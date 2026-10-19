"""CLI evaluation harness running benchmarks across configs, logging JSONL results."""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.config import settings
from app.db import execute_query, get_db_path
from app.graph import run as run_agent
from app.prompts import get_prompts_hash
from app.state import AgentConfig
from app.validate import validate
from eval.metrics import (
    calculate_cost,
    execution_match,
    extract_gold_tables_and_columns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Master Configuration Matrix
CONFIGS: Dict[str, AgentConfig] = {
    "C0": {
        "use_retrieval": False,
        "hybrid": False,
        "use_fewshot": False,
        "use_repair": False,
        "max_attempts": 1,
    },
    "C1": {
        "use_retrieval": True,
        "hybrid": False,
        "use_fewshot": False,
        "use_repair": False,
        "max_attempts": 1,
    },
    "C2": {
        "use_retrieval": True,
        "hybrid": True,
        "use_fewshot": False,
        "use_repair": False,
        "max_attempts": 1,
    },
    "C3": {
        "use_retrieval": True,
        "hybrid": True,
        "use_fewshot": True,
        "use_repair": False,
        "max_attempts": 1,
    },
    "C4": {
        "use_retrieval": True,
        "hybrid": True,
        "use_fewshot": True,
        "use_repair": True,
        "max_attempts": 3,
    },
    "L1": {  # C4 without retrieval (ablation)
        "use_retrieval": False,
        "hybrid": False,
        "use_fewshot": True,
        "use_repair": True,
        "max_attempts": 3,
    },
    "L2": {  # C4 without few-shot (ablation)
        "use_retrieval": True,
        "hybrid": True,
        "use_fewshot": False,
        "use_repair": True,
        "max_attempts": 3,
    },
    "L3": {  # C4 without repair (ablation)
        "use_retrieval": True,
        "hybrid": True,
        "use_fewshot": True,
        "use_repair": False,
        "max_attempts": 1,
    },
}


def load_subset_items(dataset: str, subset: str) -> List[Dict[str, Any]]:
    """Load items for the requested dataset and subset name."""
    subsets_dir = Path("./eval/subsets")

    if subset in ["eval200", "dev200", "eval"]:
        fname = f"{dataset.lower()}_dev_200.json" if dataset.lower() == "spider" else "bird_mini_200.json"
    elif subset in ["tune50", "tune", "tuning"]:
        fname = f"{dataset.lower()}_tune_50.json"
    else:
        # Fallback to direct filename match
        fname = f"{subset}.json" if not subset.endswith(".json") else subset

    path = subsets_dir / fname
    if not path.exists():
        # Check if generic tune_50 requested
        alt_path = subsets_dir / "tune_50.json"
        if alt_path.exists() and "tune" in subset:
            path = alt_path
        else:
            raise FileNotFoundError(f"Subset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter by dataset if loading combined
    if dataset:
        filtered = [it for it in data if it.get("dataset", dataset) == dataset]
        if filtered:
            return filtered
    return data


def evaluate_single_question(
    item: Dict[str, Any],
    config_name: str,
    config: AgentConfig,
    catalog_mode: bool = False,
) -> Dict[str, Any]:
    """Execute evaluation for one benchmark question."""
    qid = item["qid"]
    gold_db_id = item["db_id"]
    question = item["question"]
    gold_sql = item["gold_sql"]
    difficulty = item.get("difficulty", "unknown")
    evidence = item.get("evidence", "")

    # In catalog mode, agent router must determine db_id; otherwise pass gold_db_id
    query_db_id = None if catalog_mode else gold_db_id

    # 1. Run agent graph
    try:
        state = run_agent(
            question=question,
            db_id=query_db_id,
            config=config,
            evidence=evidence,
        )
    except Exception as exc:
        logger.error("Agent run failed on question %s: %s", qid, exc)
        return {
            "qid": qid,
            "db_id": gold_db_id,
            "difficulty": difficulty,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": "",
            "correct": False,
            "multiset_match": False,
            "valid_sql": False,
            "attempts": 1,
            "status": "failed",
            "error_type": f"AGENT_RUN_ERROR: {str(exc)[:100]}",
            "tokens_in": 0,
            "tokens_out": 0,
            "llm_calls": 1,
            "latency_s": 0.0,
            "cost_usd": 0.0,
            "retrieved_tables": [],
            "gold_tables": [],
            "table_recall": 0.0,
            "column_recall": 0.0,
            "schema_truncated": False,
        }

    pred_sql = state.get("sql", "").strip()
    attempts = state.get("attempts", 1)
    metrics = state.get("metrics") or {}
    tokens_in = metrics.get("tokens_in", 0)
    tokens_out = metrics.get("tokens_out", 0)
    llm_calls = metrics.get("llm_calls", 1)
    latency_s = metrics.get("total_latency_s", 0.0)

    model_name = settings.llm_model
    cost_usd = calculate_cost(model_name, tokens_in, tokens_out)

    retrieved_tables = list(state.get("tables") or [])
    retrieved_columns = list(state.get("columns") or [])
    schema_truncated = state.get("schema_truncated", False)

    # 2. Extract gold tables and columns for retrieval metrics
    gold_tables_set, gold_cols_set = extract_gold_tables_and_columns(gold_sql)
    ret_tbl_set = {t.lower() for t in retrieved_tables}
    ret_col_set = {c.lower() for c in retrieved_columns}

    table_recall = (
        len(ret_tbl_set & gold_tables_set) / len(gold_tables_set)
        if gold_tables_set
        else 1.0
    )
    column_recall = (
        len(ret_col_set & gold_cols_set) / len(gold_cols_set)
        if gold_cols_set
        else 1.0
    )

    # 3. Check AST validity
    val_res = validate(pred_sql)
    valid_sql = val_res.ok

    # 4. Execute queries for execution accuracy
    correct = False
    multiset_match = False
    error_type = None

    try:
        db_path = get_db_path(gold_db_id)
    except Exception:
        return {
            "qid": qid,
            "db_id": gold_db_id,
            "difficulty": difficulty,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "correct": False,
            "multiset_match": False,
            "valid_sql": valid_sql,
            "attempts": attempts,
            "status": "failed",
            "error_type": "DB_NOT_FOUND",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "llm_calls": llm_calls,
            "latency_s": latency_s,
            "cost_usd": cost_usd,
            "retrieved_tables": retrieved_tables,
            "gold_tables": list(gold_tables_set),
            "table_recall": table_recall,
            "column_recall": column_recall,
            "schema_truncated": schema_truncated,
        }

    # Execute Gold SQL
    gold_timeout_s = settings.eval_query_timeout_s
    try:
        gold_res = execute_query(db_path, gold_sql, timeout_s=gold_timeout_s, max_rows=None)
    except Exception as exc:
        logger.warning("Gold SQL failed/timed out on %s [%s]: %s", qid, gold_db_id, exc)
        return {
            "qid": qid,
            "db_id": gold_db_id,
            "difficulty": difficulty,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "correct": False,
            "multiset_match": False,
            "valid_sql": valid_sql,
            "attempts": attempts,
            "status": "failed",
            "error_type": "GOLD_SQL_ERROR",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "llm_calls": llm_calls,
            "latency_s": latency_s,
            "cost_usd": cost_usd,
            "retrieved_tables": retrieved_tables,
            "gold_tables": list(gold_tables_set),
            "table_recall": table_recall,
            "column_recall": column_recall,
            "schema_truncated": schema_truncated,
        }

    # Execute Pred SQL
    if not valid_sql:
        error_type = val_res.error_type or "INVALID_SQL"
    else:
        try:
            pred_res = execute_query(db_path, pred_sql, timeout_s=gold_timeout_s, max_rows=None)
            s_match, m_match = execution_match(pred_res.rows, gold_res.rows)
            correct = s_match
            multiset_match = m_match

            if not correct:
                if len(pred_res.rows) == 0 and len(gold_res.rows) > 0:
                    error_type = "EMPTY_RESULT"
                elif not (gold_tables_set.issubset({t.lower() for t in extract_gold_tables_and_columns(pred_sql)[0]})):
                    error_type = "WRONG_TABLE"
                else:
                    error_type = "WRONG_LOGIC_OR_COLUMNS"
        except Exception:
            correct = False
            error_type = "EXECUTION_ERROR"

    return {
        "qid": qid,
        "db_id": gold_db_id,
        "difficulty": difficulty,
        "question": question,
        "gold_sql": gold_sql,
        "pred_sql": pred_sql,
        "correct": correct,
        "multiset_match": multiset_match,
        "valid_sql": valid_sql,
        "attempts": attempts,
        "status": "ok" if correct else "failed",
        "error_type": error_type,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "llm_calls": llm_calls,
        "latency_s": latency_s,
        "cost_usd": cost_usd,
        "retrieved_tables": retrieved_tables,
        "gold_tables": list(gold_tables_set),
        "table_recall": table_recall,
        "column_recall": column_recall,
        "schema_truncated": schema_truncated,
    }


def run_evaluation(
    dataset: str,
    config_name: str,
    subset: str = "eval200",
    mode: str = "db",
    workers: int = 1,
    output_dir: str = "./eval/results",
    limit: Optional[int] = None,
) -> Path:
    """Run evaluation and stream results to JSONL."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Must be one of: {list(CONFIGS.keys())}")

    config = dict(CONFIGS[config_name])
    catalog_mode = (mode == "catalog")
    config["catalog_mode"] = catalog_mode

    items = load_subset_items(dataset, subset)
    if limit is not None and limit > 0:
        items = items[:limit]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{dataset}_{subset}_{config_name}_{mode}.jsonl"

    completed_qids: Set[str] = set()
    file_exists = out_file.exists()

    # Resume support: check finished questions
    if file_exists:
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                    if "qid" in obj:
                        completed_qids.add(obj["qid"])
                except Exception:
                    pass
        logger.info("Resuming %s: %d questions already completed.", out_file.name, len(completed_qids))
    else:
        # Write header line
        import importlib.metadata

        def get_ver(pkg: str) -> str:
            try:
                return importlib.metadata.version(pkg)
            except Exception:
                return "unknown"

        header = {
            "header": True,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "run_date": datetime.now(timezone.utc).isoformat(),
            "library_versions": {
                "langchain": get_ver("langchain"),
                "langgraph": get_ver("langgraph"),
                "sqlglot": get_ver("sqlglot"),
                "chromadb": get_ver("chromadb"),
            },
            "prompt_hash": get_prompts_hash(),
            "seed": 42,
            "config_name": config_name,
            "config": config,
            "dataset": dataset,
            "subset": subset,
            "mode": mode,
        }
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")

    remaining_items = [it for it in items if it["qid"] not in completed_qids]
    logger.info(
        "Starting eval: dataset=%s, subset=%s, config=%s, mode=%s | %d/%d to run",
        dataset,
        subset,
        config_name,
        mode,
        len(remaining_items),
        len(items),
    )

    # Process items
    num_workers = max(1, workers)
    if num_workers == 1:
        for idx, item in enumerate(remaining_items, 1):
            logger.info("[%d/%d] Running %s (%s)...", idx, len(remaining_items), item["qid"], item["db_id"])
            rec = evaluate_single_question(item, config_name, config, catalog_mode=catalog_mode)
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {
                executor.submit(
                    evaluate_single_question, item, config_name, config, catalog_mode
                ): item
                for item in remaining_items
            }
            completed_cnt = 0
            for future in as_completed(future_to_item):
                completed_cnt += 1
                rec = future.result()
                logger.info(
                    "[%d/%d] Finished %s (correct=%s, attempts=%d)",
                    completed_cnt,
                    len(remaining_items),
                    rec["qid"],
                    rec["correct"],
                    rec["attempts"],
                )
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

    logger.info("Evaluation finished. Output saved to %s", out_file)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Text-to-SQL benchmark evaluation.")
    parser.add_argument("--dataset", required=True, choices=["spider", "bird"], help="Dataset name")
    parser.add_argument(
        "--config",
        required=True,
        choices=["C0", "C1", "C2", "C3", "C4", "L1", "L2", "L3"],
        help="Configuration ID",
    )
    parser.add_argument("--subset", default="eval200", help="Subset name (eval200 or tune50)")
    parser.add_argument("--mode", default="db", choices=["db", "catalog"], help="db or catalog mode")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent worker threads")
    parser.add_argument("--output-dir", default="./eval/results", help="Directory for JSONL outputs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for testing)")

    args = parser.parse_args()
    run_evaluation(
        dataset=args.dataset,
        config_name=args.config,
        subset=args.subset,
        mode=args.mode,
        workers=args.workers,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
