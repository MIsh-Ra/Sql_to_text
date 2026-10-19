"""Evaluation metrics: execution match, bootstrap CIs, paired tests, and costs."""

import collections
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import sqlglot
import yaml
from scipy import stats
from sqlglot import exp

logger = logging.getLogger(__name__)


def _normalize_cell_value(val: Any) -> Any:
    """Normalize a single cell value for robust execution matching."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return "NaN"
        if np.isinf(val):
            return "Inf"
        return round(float(val), 4)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, str):
        # Strip trailing/leading spaces and normalize whitespace
        val_str = val.strip()
        # Try casting to float if string is numeric
        try:
            val_float = float(val_str)
            return round(val_float, 4)
        except ValueError:
            return val_str
    return str(val)


def _normalize_row(row: Iterable[Any]) -> Tuple[Any, ...]:
    """Normalize a row tuple."""
    return tuple(_normalize_cell_value(c) for c in row)


def execution_match(
    pred_rows: Optional[List[Tuple[Any, ...]]],
    gold_rows: Optional[List[Tuple[Any, ...]]],
) -> Tuple[bool, bool]:
    """Compare predicted rows against gold rows.

    Returns:
        (set_match, multiset_match)
        - set_match: compares sets of normalized row tuples.
        - multiset_match: stricter multiset (Counter) match comparing frequencies.
    """
    if pred_rows is None or gold_rows is None:
        return False, False

    try:
        norm_pred = [_normalize_row(r) for r in pred_rows]
        norm_gold = [_normalize_row(r) for r in gold_rows]

        # 1. Set match
        set_pred = set(norm_pred)
        set_gold = set(norm_gold)
        set_matched = (set_pred == set_gold)

        # 2. Multiset match
        counter_pred = collections.Counter(norm_pred)
        counter_gold = collections.Counter(norm_gold)
        multiset_matched = (counter_pred == counter_gold)

        return set_matched, multiset_matched
    except Exception as exc:
        logger.debug("Execution match comparison raised exception: %s", exc)
        return False, False


def extract_gold_tables_and_columns(gold_sql: str) -> Tuple[Set[str], Set[str]]:
    """Parse gold SQL using sqlglot and extract referenced tables and columns."""
    clean_sql = gold_sql.strip().rstrip(";")
    tables: Set[str] = set()
    columns: Set[str] = set()

    try:
        ast = sqlglot.parse_one(clean_sql, read="sqlite")
        for tbl in ast.find_all(exp.Table):
            if tbl.name:
                tables.add(tbl.name.lower())
        for col in ast.find_all(exp.Column):
            if col.name and col.name != "*":
                columns.add(col.name.lower())
    except Exception as exc:
        logger.debug("Could not parse gold SQL for table/column extraction: %s", exc)

    return tables, columns


def bootstrap_ci(
    scores: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate mean and bootstrap confidence interval (percentile method).

    Returns:
        (mean_score, ci_lower, ci_upper)
    """
    if not scores:
        return 0.0, 0.0, 0.0

    arr = np.array(scores, dtype=np.float64)
    mean_val = float(np.mean(arr))
    n = len(arr)

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(arr[indices])

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return mean_val, ci_lower, ci_upper


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Paired bootstrap test for difference in means (A - B).

    Returns:
        (mean_diff, ci_lower, ci_upper, p_value)
    """
    assert len(scores_a) == len(scores_b), "Paired tests require equal length score lists"
    n = len(scores_a)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0

    diffs = np.array(scores_a, dtype=np.float64) - np.array(scores_b, dtype=np.float64)
    mean_diff = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    boot_diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_diffs[i] = np.mean(diffs[indices])

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Two-sided empirical p-value for null hypothesis mean_diff == 0
    p_value = float(np.mean(boot_diffs <= 0.0) if mean_diff > 0 else np.mean(boot_diffs >= 0.0)) * 2.0
    p_value = min(1.0, max(0.0, p_value))

    return mean_diff, ci_lower, ci_upper, p_value


def exact_mcnemar_test(
    correct_a: List[bool],
    correct_b: List[bool],
) -> Tuple[int, int, float]:
    """Exact McNemar test using binomial distribution.

    Contingency table cells:
        b: A incorrect (False), B correct (True)
        c: A correct (True), B incorrect (False)

    Returns:
        (b, c, p_value)
    """
    assert len(correct_a) == len(correct_b), "Paired tests require matching question sets"

    b = sum(1 for a, b_val in zip(correct_a, correct_b) if (not a) and b_val)
    c = sum(1 for a, b_val in zip(correct_a, correct_b) if a and (not b_val))

    total = b + c
    if total == 0:
        return 0, 0, 1.0

    # Exact binomial test with p = 0.5
    res = stats.binomtest(min(b, c), total, 0.5, alternative="two-sided")
    return b, c, float(res.pvalue)


def calculate_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
    pricing_path: Optional[Path] = None,
) -> float:
    """Calculate cost in USD based on eval/pricing.yaml."""
    path = pricing_path or Path("./eval/pricing.yaml")
    rates = {"input_per_m": 0.15, "output_per_m": 0.60}

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            models = data.get("models", {})
            if model in models:
                rates = models[model]
            elif "default" in data:
                rates = data["default"]
        except Exception:
            pass

    cost_in = (tokens_in / 1_000_000.0) * float(rates.get("input_per_m", 0.15))
    cost_out = (tokens_out / 1_000_000.0) * float(rates.get("output_per_m", 0.60))
    return cost_in + cost_out


def compute_aggregate_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate evaluation summary across an eval run."""
    if not records:
        return {}

    n = len(records)
    correct_list = [1.0 if r.get("correct", False) else 0.0 for r in records]
    valid_list = [1.0 if r.get("valid_sql", False) else 0.0 for r in records]
    multiset_list = [1.0 if r.get("multiset_match", False) else 0.0 for r in records]

    acc_mean, acc_low, acc_high = bootstrap_ci(correct_list)
    val_mean, _, _ = bootstrap_ci(valid_list)
    multi_mean, _, _ = bootstrap_ci(multiset_list)

    latencies = [r.get("latency_s", 0.0) for r in records]
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    tokens_in = [r.get("tokens_in", 0) for r in records]
    tokens_out = [r.get("tokens_out", 0) for r in records]
    total_cost = sum(r.get("cost_usd", 0.0) for r in records)
    attempts = [r.get("attempts", 1) for r in records]

    # Retrieval metrics
    tbl_recalls = [r.get("table_recall", 1.0) for r in records if "table_recall" in r]
    col_recalls = [r.get("column_recall", 1.0) for r in records if "column_recall" in r]

    # Per-difficulty breakdown
    by_diff: Dict[str, List[float]] = collections.defaultdict(list)
    for r in records:
        diff = r.get("difficulty", "unknown")
        by_diff[diff].append(1.0 if r.get("correct", False) else 0.0)

    diff_summary = {}
    for diff, scs in by_diff.items():
        m, lo, hi = bootstrap_ci(scs)
        diff_summary[diff] = {"count": len(scs), "accuracy": m, "ci_lower": lo, "ci_upper": hi}

    return {
        "n_samples": n,
        "execution_accuracy": acc_mean,
        "ci_lower": acc_low,
        "ci_upper": acc_high,
        "multiset_accuracy": multi_mean,
        "valid_sql_rate": val_mean,
        "mean_attempts": float(np.mean(attempts)) if attempts else 1.0,
        "p50_latency_s": p50_lat,
        "p95_latency_s": p95_lat,
        "mean_tokens_in": float(np.mean(tokens_in)) if tokens_in else 0.0,
        "mean_tokens_out": float(np.mean(tokens_out)) if tokens_out else 0.0,
        "total_cost_usd": total_cost,
        "mean_table_recall": float(np.mean(tbl_recalls)) if tbl_recalls else 1.0,
        "mean_column_recall": float(np.mean(col_recalls)) if col_recalls else 1.0,
        "difficulty_breakdown": diff_summary,
    }

