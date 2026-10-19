"""Build schema and few-shot vector indexes and compute retrieval benchmarks."""

import json
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import full_schema
from app.fewshot_index import get_fewshot_index
from app.schema_index import get_schema_index, retrieve_schema
from eval.metrics import extract_gold_tables_and_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_all_indexes() -> None:
    logger.info("Initializing Index Building...")

    schema_idx = get_schema_index()
    fs_idx = get_fewshot_index()

    # 1. Build Few-Shot Index from Spider Train
    logger.info("Building Few-Shot Index from Spider Train...")
    fs_count = fs_idx.build_from_train(max_samples=1000)
    logger.info("Few-Shot Index ready with %d examples.", fs_count)

    # 2. Index Spider Dev Databases (20 databases)
    spider_db_dir = Path("./data/spider/database")
    if spider_db_dir.exists():
        spider_dbs = [d.name for d in spider_db_dir.iterdir() if d.is_dir()]
        logger.info("Indexing %d Spider dev databases...", len(spider_dbs))
        for db in spider_dbs:
            try:
                schema_idx.index_database(db)
            except Exception as exc:
                logger.warning("Error indexing Spider db %s: %s", db, exc)

    # 3. Index BIRD Dev Databases (11 databases)
    bird_db_dir = Path("./data/bird/dev_databases")
    if bird_db_dir.exists():
        bird_dbs = [d.name for d in bird_db_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        logger.info("Indexing %d BIRD dev databases...", len(bird_dbs))
        for db in bird_dbs:
            try:
                schema_idx.index_database(db)
            except Exception as exc:
                logger.warning("Error indexing BIRD db %s: %s", db, exc)

    logger.info("All databases indexed into Chroma.")


def evaluate_retrieval_performance(subset_path: Path, dataset_name: str) -> None:
    """Evaluate Table Recall@k and Token Reduction on an evaluation subset."""
    with open(subset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    logger.info("Evaluating retrieval on %s (%d questions)...", subset_path.name, len(items))

    table_recalls: list[float] = []
    full_tokens_list: list[int] = []
    retrieved_tokens_list: list[int] = []

    for idx, item in enumerate(items, 1):
        q = item["question"]
        db = item["db_id"]
        gold_sql = item["gold_sql"]

        gold_tables, _ = extract_gold_tables_and_columns(gold_sql)

        # Full schema baseline
        full_text, _, _, _ = full_schema(db)
        full_tokens = max(1, len(full_text) // 4)
        full_tokens_list.append(full_tokens)

        # Retrieved schema (hybrid)
        ret_text, ret_tables, ret_cols, _ = retrieve_schema(
            question=q, db_id=db, k_tables=6, k_columns=25, hybrid=True
        )
        ret_tokens = max(1, len(ret_text) // 4)
        retrieved_tokens_list.append(ret_tokens)

        # Table Recall
        ret_tbl_set = {t.lower() for t in ret_tables}
        rec = len(ret_tbl_set & gold_tables) / len(gold_tables) if gold_tables else 1.0
        table_recalls.append(rec)

    mean_recall = sum(table_recalls) / len(table_recalls) if table_recalls else 0.0
    mean_full_tok = sum(full_tokens_list) / len(full_tokens_list) if full_tokens_list else 1
    mean_ret_tok = sum(retrieved_tokens_list) / len(retrieved_tokens_list) if retrieved_tokens_list else 1
    reduction_pct = (1.0 - (mean_ret_tok / mean_full_tok)) * 100.0

    print("\n" + "=" * 60)
    print(f"RETRIEVAL BENCHMARK: {dataset_name.upper()} ({len(items)} questions)")
    print("=" * 60)
    print(f"Mean Table Recall@k:       {mean_recall * 100.0:.2f}%")
    print(f"Mean Baseline Tokens:      {mean_full_tok:.1f}")
    print(f"Mean Retrieved Tokens:     {mean_ret_tok:.1f}")
    print(f"Token Reduction:           {reduction_pct:.2f}%")
    print("=" * 60 + "\n")


def main() -> None:
    build_all_indexes()

    spider_eval = Path("./eval/subsets/spider_dev_200.json")
    if spider_eval.exists():
        evaluate_retrieval_performance(spider_eval, "Spider Dev 200")

    bird_eval = Path("./eval/subsets/bird_mini_200.json")
    if bird_eval.exists():
        evaluate_retrieval_performance(bird_eval, "BIRD Mini 200")


if __name__ == "__main__":
    main()

