"""Aggregate benchmark evaluation JSONL runs into eval/results.md and eval/results.png."""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (
    bootstrap_ci,
    exact_mcnemar_test,
    paired_bootstrap_test,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_run_file(file_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse a single evaluation JSONL file into header and question records."""
    header = {}
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            if idx == 0 and data.get("header"):
                header = data
            else:
                records.append(data)

    return header, records


def validate_run_compatibility(headers: List[Dict[str, Any]]) -> None:
    """Verify that runs share the identical model and prompt hash."""
    if not headers:
        return

    reference_model = headers[0].get("model")
    reference_hash = headers[0].get("prompt_hash")

    for h in headers[1:]:
        m = h.get("model")
        ph = h.get("prompt_hash")
        if m != reference_model:
            raise ValueError(
                f"Cannot aggregate runs with different models: '{reference_model}' vs '{m}'. "
                "All benchmark comparisons must be executed with identical models."
            )
        if ph != reference_hash:
            raise ValueError(
                f"Cannot aggregate runs with different prompt hashes: '{reference_hash}' vs '{ph}'. "
                "Prompts must be frozen across all configurations."
            )


def generate_accuracy_chart(
    runs_by_dataset: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Path,
) -> None:
    """Generate professional PNG bar chart with 95% bootstrap confidence interval error bars."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    config_order = ["C0", "C1", "C2", "C3", "C4", "L1", "L2", "L3"]
    datasets = ["spider", "bird"]
    colors = {"spider": "#2563eb", "bird": "#059669"}
    labels = {"spider": "Spider Dev (200 Qs)", "bird": "BIRD Mini-Dev (200 Qs)"}

    x = np.arange(len(config_order))
    width = 0.38

    for i, ds in enumerate(datasets):
        ds_runs = runs_by_dataset.get(ds, {})
        means = []
        err_low = []
        err_high = []

        for cfg in config_order:
            if cfg in ds_runs:
                stats = ds_runs[cfg]
                m = stats["accuracy"] * 100.0
                lo = (stats["accuracy"] - stats["ci_lower"]) * 100.0
                hi = (stats["ci_upper"] - stats["accuracy"]) * 100.0
                means.append(m)
                err_low.append(max(0.0, lo))
                err_high.append(max(0.0, hi))
            else:
                means.append(0.0)
                err_low.append(0.0)
                err_high.append(0.0)

        offset = width * (i - 0.5)
        rects = ax.bar(
            x + offset,
            means,
            width,
            yerr=[err_low, err_high],
            capsize=4,
            label=labels[ds],
            color=colors[ds],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_ylabel("Execution Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Text-to-SQL Agent: Ablation Study & Execution Accuracy (95% Bootstrap CIs)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(config_order, fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)
    ax.set_ylim(0, 100)

    # Note on config definitions below axis
    note_text = (
        "C0: Baseline (Single-Prompt) | C1: Dense Schema | C2: Hybrid Schema (RRF k=60) | "
        "C3: Few-Shot | C4: Full Agent (Repair Loop)\n"
        "L1: C4 without Retrieval | L2: C4 without Few-Shot | L3: C4 without Repair"
    )
    plt.figtext(0.5, 0.01, note_text, ha="center", fontsize=8, color="#444", style="italic")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved accuracy comparison chart to %s", output_path)


def aggregate_results(results_dir: Path = Path("./eval/results")) -> Path:
    """Aggregate all evaluation run JSONL files into eval/results.md and eval/results.png."""
    jsonl_files = list(results_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No evaluation JSONL results found in {results_dir}")

    runs_data: List[Dict[str, Any]] = []
    headers: List[Dict[str, Any]] = []

    for fpath in jsonl_files:
        header, records = parse_run_file(fpath)
        if not header or not records:
            continue
        # Strictly evaluate eval200 benchmark files to prevent tune sets from overwriting results
        if header.get("subset") != "eval200":
            logger.info("Skipping non-eval200 run: %s (subset=%s)", fpath.name, header.get("subset"))
            continue
        headers.append(header)
        runs_data.append({
            "file": fpath.name,
            "header": header,
            "records": records,
            "dataset": header.get("dataset"),
            "subset": header.get("subset"),
            "config_name": header.get("config_name"),
            "mode": header.get("mode"),
        })

    # Validate compatibility (same model & prompt hash)
    validate_run_compatibility(headers)

    model_name = headers[0].get("model", "unknown")
    provider = headers[0].get("provider", "unknown")
    prompt_hash = headers[0].get("prompt_hash", "unknown")
    run_date = headers[0].get("run_date", "")[:10]

    # Organize runs by (dataset, config_name)
    processed_runs: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    records_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for run in runs_data:
        ds = run["dataset"]
        cfg = run["config_name"]
        records = run["records"]
        records_by_key[(ds, cfg)] = records

        scores = [1.0 if r.get("correct") else 0.0 for r in records]
        val_scores = [1.0 if r.get("valid_sql") else 0.0 for r in records]
        multi_scores = [1.0 if r.get("multiset_match") else 0.0 for r in records]

        acc_mean, acc_low, acc_high = bootstrap_ci(scores)
        val_mean, _, _ = bootstrap_ci(val_scores)
        multi_mean, _, _ = bootstrap_ci(multi_scores)

        attempts = [r.get("attempts", 1) for r in records]
        latencies = [r.get("latency_s", 0.0) for r in records]
        tok_in = [r.get("tokens_in", 0) for r in records]
        tok_out = [r.get("tokens_out", 0) for r in records]
        costs = [r.get("cost_usd", 0.0) for r in records]
        tbl_rec = [r.get("table_recall", 1.0) for r in records]
        col_rec = [r.get("column_recall", 1.0) for r in records]

        # Per difficulty
        diff_map: Dict[str, List[float]] = defaultdict(list)
        for r in records:
            d = r.get("difficulty", "unknown")
            diff_map[d].append(1.0 if r.get("correct") else 0.0)

        diff_stats = {}
        for d, scs in diff_map.items():
            dm, dl, dh = bootstrap_ci(scs)
            diff_stats[d] = {"mean": dm, "low": dl, "high": dh, "count": len(scs)}

        processed_runs[ds][cfg] = {
            "n": len(records),
            "accuracy": acc_mean,
            "ci_lower": acc_low,
            "ci_upper": acc_high,
            "multiset_accuracy": multi_mean,
            "valid_sql_rate": val_mean,
            "mean_attempts": float(np.mean(attempts)),
            "p50_latency_s": float(np.percentile(latencies, 50)),
            "p95_latency_s": float(np.percentile(latencies, 95)),
            "mean_tokens_in": float(np.mean(tok_in)),
            "mean_tokens_out": float(np.mean(tok_out)),
            "total_cost_usd": sum(costs),
            "mean_table_recall": float(np.mean(tbl_rec)),
            "mean_column_recall": float(np.mean(col_rec)),
            "difficulties": diff_stats,
        }

    # Generate Chart
    chart_path = Path("./eval/results.png")
    generate_accuracy_chart(processed_runs, chart_path)

    # Build Markdown Document
    md_lines = []
    md_lines.append("# Empirical Benchmark Results\n")
    md_lines.append(f"**Evaluated Model**: `{model_name}` ({provider})  ")
    md_lines.append(f"**Evaluation Date**: `{run_date}`  ")
    md_lines.append(f"**Frozen Prompt SHA256**: `{prompt_hash}`  ")
    md_lines.append("**Sampling Seed**: `42` | **Temperature**: `0.0`\n")

    # 1. Headline Results Table (C0 vs C4)
    md_lines.append("## Headline Results: Baseline (C0) vs Full Agent (C4)\n")
    md_lines.append("| Benchmark Subset | C0 Baseline (Acc & 95% CI) | C4 Full Agent (Acc & 95% CI) | Delta (pts) | McNemar p-value | Valid SQL Rate |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for ds, name in [("spider", "Spider Dev (200 Qs)"), ("bird", "BIRD Mini-Dev (200 Qs)")]:
        c0 = processed_runs.get(ds, {}).get("C0")
        c4 = processed_runs.get(ds, {}).get("C4")
        if c0 and c4:
            # Paired significance
            recs_0 = {r["qid"]: r for r in records_by_key.get((ds, "C0"), [])}
            recs_4 = {r["qid"]: r for r in records_by_key.get((ds, "C4"), [])}
            common_qids = sorted(list(set(recs_0.keys()) & set(recs_4.keys())))

            corr_0 = [recs_0[q]["correct"] for q in common_qids]
            corr_4 = [recs_4[q]["correct"] for q in common_qids]
            _, _, p_mcnemar = exact_mcnemar_test(corr_0, corr_4)

            delta = (c4["accuracy"] - c0["accuracy"]) * 100.0
            sign = "+" if delta >= 0 else ""
            c0_str = f"{c0['accuracy']*100:.1f}% [{c0['ci_lower']*100:.1f}%, {c0['ci_upper']*100:.1f}%]"
            c4_str = f"{c4['accuracy']*100:.1f}% [{c4['ci_lower']*100:.1f}%, {c4['ci_upper']*100:.1f}%]"
            val_str = f"{c0['valid_sql_rate']*100:.1f}% -> {c4['valid_sql_rate']*100:.1f}%"
            p_str = f"{p_mcnemar:.4f}" if p_mcnemar >= 0.0001 else "< 0.0001"
            md_lines.append(f"| **{name}** | {c0_str} | **{c4_str}** | **{sign}{delta:.1f}** | {p_str} | {val_str} |")

    md_lines.append("\n![Benchmark Accuracy](results.png)\n")

    # 2. Comprehensive Ablation Table
    md_lines.append("## Complete Ablation Study\n")
    md_lines.append(
        "| Config | Description | Spider Dev (Acc & CI) | BIRD Mini (Acc & CI) | Valid SQL | Mean Attempts | Mean Tokens (In / Out) | p50 / p95 Latency |"
    )
    md_lines.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )

    config_descriptions = {
        "C0": "Naive Single-Prompt Baseline (No retrieval, no few-shots, no repair)",
        "C1": "Dense Schema Retrieval (BAAI/bge-small-en-v1.5)",
        "C2": "Hybrid Schema Retrieval (Dense + BM25, RRF k=60)",
        "C3": "Few-Shot Retrieval (k=3 demonstrations, strict db exclusion)",
        "C4": "Full Agent (Hybrid Schema + Few-Shot + AST Validation + Repair)",
        "L1": "Ablation: C4 without Schema Retrieval",
        "L2": "Ablation: C4 without Few-Shot Injection",
        "L3": "Ablation: C4 without Error Repair Loop",
    }

    for cfg in ["C0", "C1", "C2", "C3", "C4", "L1", "L2", "L3"]:
        desc = config_descriptions.get(cfg, "")
        sp_stat = processed_runs.get("spider", {}).get(cfg)
        bd_stat = processed_runs.get("bird", {}).get(cfg)

        sp_acc = f"{sp_stat['accuracy']*100:.1f}% [{sp_stat['ci_lower']*100:.1f}%, {sp_stat['ci_upper']*100:.1f}%]" if sp_stat else "N/A"
        bd_acc = f"{bd_stat['accuracy']*100:.1f}% [{bd_stat['ci_lower']*100:.1f}%, {bd_stat['ci_upper']*100:.1f}%]" if bd_stat else "N/A"

        primary = sp_stat or bd_stat
        if primary:
            val_rate = f"{primary['valid_sql_rate']*100:.1f}%"
            att = f"{primary['mean_attempts']:.2f}"
            tok = f"{int(primary['mean_tokens_in'])} / {int(primary['mean_tokens_out'])}"
            lat = f"{primary['p50_latency_s']:.1f}s / {primary['p95_latency_s']:.1f}s"
        else:
            val_rate = att = tok = lat = "N/A"

        md_lines.append(f"| **{cfg}** | {desc} | {sp_acc} | {bd_acc} | {val_rate} | {att} | {tok} | {lat} |")

    # 3. Paired Significance Tests Table
    md_lines.append("\n## Statistical Significance Tests (Paired on Matching Question IDs)\n")
    md_lines.append("| Comparison | Dataset | Paired Mean Diff (pts) | 95% Bootstrap CI | Discordant Pairs (b / c) | McNemar p-value | Significant (p < 0.05) |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    comparisons = [
        ("C0 vs C4 (Full Agent Gain)", "C0", "C4"),
        ("C0 vs C2 (Hybrid Retrieval Effect)", "C0", "C2"),
        ("C2 vs C3 (Few-Shot Effect)", "C2", "C3"),
        ("C3 vs C4 (Repair Loop Effect)", "C3", "C4"),
        ("C4 vs L1 (Retrieval Ablation)", "L1", "C4"),
        ("C4 vs L2 (Few-Shot Ablation)", "L2", "C4"),
        ("C4 vs L3 (Repair Ablation)", "L3", "C4"),
    ]

    for label, base_cfg, test_cfg in comparisons:
        for ds in ["spider", "bird"]:
            recs_base = {r["qid"]: r for r in records_by_key.get((ds, base_cfg), [])}
            recs_test = {r["qid"]: r for r in records_by_key.get((ds, test_cfg), [])}
            common = sorted(list(set(recs_base.keys()) & set(recs_test.keys())))
            if len(common) > 10:
                sc_base = [1.0 if recs_base[q]["correct"] else 0.0 for q in common]
                sc_test = [1.0 if recs_test[q]["correct"] else 0.0 for q in common]
                diff, lo, hi, _ = paired_bootstrap_test(sc_test, sc_base)
                b, c, p_mcnemar = exact_mcnemar_test(
                    [bool(x) for x in sc_base], [bool(x) for x in sc_test]
                )
                sig = "Yes" if p_mcnemar < 0.05 else "No"
                p_disp = f"{p_mcnemar:.4f}" if p_mcnemar >= 0.0001 else "< 0.0001"
                md_lines.append(
                    f"| {label} | {ds.capitalize()} | {diff*100:+.1f} | [{lo*100:+.1f}, {hi*100:+.1f}] | b={b}, c={c} | {p_disp} | **{sig}** |"
                )

    # 4. Per-Difficulty Breakdown Table
    md_lines.append("\n## Accuracy Breakdown by Difficulty\n")
    md_lines.append("| Dataset | Difficulty Tier | Sample Count | Baseline C0 Acc | Full Agent C4 Acc | Relative Gain |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for ds in ["spider", "bird"]:
        c0_diffs = processed_runs.get(ds, {}).get("C0", {}).get("difficulties", {})
        c4_diffs = processed_runs.get(ds, {}).get("C4", {}).get("difficulties", {})
        all_diffs = sorted(list(set(c0_diffs.keys()) | set(c4_diffs.keys())))

        for d in all_diffs:
            s0 = c0_diffs.get(d, {"mean": 0.0, "count": 0})
            s4 = c4_diffs.get(d, {"mean": 0.0, "count": 0})
            count = max(s0["count"], s4["count"])
            g0 = s0["mean"] * 100.0
            g4 = s4["mean"] * 100.0
            gain = g4 - g0
            md_lines.append(
                f"| {ds.capitalize()} | **{d.capitalize()}** | {count} | {g0:.1f}% | **{g4:.1f}%** | {gain:+.1f} pts |"
            )

    # 5. Retrieval & Catalog Mode Table
    md_lines.append("\n## Schema Retrieval Performance\n")
    md_lines.append("| Benchmark Subset | Retrieval Mode | Table Recall@k | Column Recall@k | Mean Schema Tokens |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- |")

    for ds, name in [("spider", "Spider Dev 200"), ("bird", "BIRD Mini 200")]:
        c2 = processed_runs.get(ds, {}).get("C2")
        if c2:
            ret_tok = c2["mean_tokens_in"]
            md_lines.append(
                f"| {name} | Hybrid (Dense + BM25, RRF) | **{c2['mean_table_recall']*100:.2f}%** | {c2['mean_column_recall']*100:.2f}% | {int(ret_tok)} tokens |"
            )

    output_md_path = Path("./eval/results.md")
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Generated %s successfully.", output_md_path)
    return output_md_path


def main() -> None:
    aggregate_results()


if __name__ == "__main__":
    main()
