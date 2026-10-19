"""Build stratified benchmark subsets for evaluation and tuning."""

import collections
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")


def _has_agg(unit: Tuple[Any, ...]) -> bool:
    return unit[0] != AGG_OPS.index("none")


def _count_agg(units: List[Any]) -> int:
    return len([unit for unit in units if _has_agg(unit)])


def _get_nested_sql(sql: Dict[str, Any]) -> List[Any]:
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
        if isinstance(cond_unit[4], dict):
            nested.append(cond_unit[4])
    if sql.get("intersect") is not None:
        nested.append(sql["intersect"])
    if sql.get("except") is not None:
        nested.append(sql["except"])
    if sql.get("union") is not None:
        nested.append(sql["union"])
    return nested


def _count_component1(sql: Dict[str, Any]) -> int:
    count = 0
    if len(sql.get("where", [])) > 0:
        count += 1
    if len(sql.get("groupBy", [])) > 0:
        count += 1
    if len(sql.get("orderBy", [])) > 0:
        count += 1
    if sql.get("limit") is not None:
        count += 1
    if len(sql.get("from", {}).get("table_units", [])) > 0:
        count += len(sql["from"]["table_units"]) - 1

    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([token for token in ao if token == "or"])
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len([cu for cu in cond_units if cu[1] == WHERE_OPS.index("like")])
    return count


def _count_component2(sql: Dict[str, Any]) -> int:
    return len(_get_nested_sql(sql))


def _count_others(sql: Dict[str, Any]) -> int:
    count = 0
    agg_count = _count_agg(sql["select"][1])
    agg_count += _count_agg(sql["where"][::2])
    agg_count += _count_agg(sql["groupBy"])
    if len(sql.get("orderBy", [])) > 0:
        agg_count += _count_agg(
            [unit[1] for unit in sql["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql["orderBy"][1] if unit[2]]
        )
    agg_count += _count_agg(sql["having"])
    if agg_count > 1:
        count += 1

    if len(sql["select"][1]) > 1:
        count += 1
    if len(sql["where"]) > 1:
        count += 1
    if len(sql["groupBy"]) > 1:
        count += 1
    return count


def compute_spider_hardness(sql: Dict[str, Any]) -> str:
    """Compute official Spider hardness label: easy, medium, hard, extra."""
    c1 = _count_component1(sql)
    c2 = _count_component2(sql)
    others = _count_others(sql)

    if c1 <= 1 and others == 0 and c2 == 0:
        return "easy"
    elif (others <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and others < 2 and c2 == 0):
        return "medium"
    elif (
        (others > 2 and c1 <= 2 and c2 == 0)
        or (2 < c1 <= 3 and others <= 2 and c2 == 0)
        or (c1 <= 1 and others == 0 and c2 <= 1)
    ):
        return "hard"
    else:
        return "extra"


def stratified_split(
    items: List[Dict[str, Any]],
    n_sample: int,
    label_key: str = "difficulty",
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified sampling by label, returning (sampled, remaining)."""
    by_label: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for it in items:
        by_label[it[label_key]].append(it)

    rng = np.random.RandomState(seed)
    total_items = len(items)
    sampled: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []

    # Calculate exact allocation per stratum
    allocated_counts = {}
    running_total = 0
    labels = sorted(by_label.keys())
    for i, label in enumerate(labels):
        group = by_label[label]
        if i == len(labels) - 1:
            k = n_sample - running_total
        else:
            k = int(round(n_sample * (len(group) / total_items)))
            running_total += k
        allocated_counts[label] = k

    for label in labels:
        group = by_label[label].copy()
        rng.shuffle(group)
        k = allocated_counts[label]
        sampled.extend(group[:k])
        remaining.extend(group[k:])

    # Final shuffle
    rng.shuffle(sampled)
    rng.shuffle(remaining)
    return sampled, remaining


def build_subsets() -> None:
    seed = 42
    out_dir = Path("./eval/subsets")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process Spider
    spider_dev_path = Path("./data/spider/dev.json")
    if not spider_dev_path.exists():
        raise FileNotFoundError(f"Missing Spider dev data at {spider_dev_path}")

    with open(spider_dev_path, "r", encoding="utf-8") as f:
        spider_dev_raw = json.load(f)

    spider_items = []
    for idx, raw in enumerate(spider_dev_raw):
        hardness = compute_spider_hardness(raw["sql"])
        spider_items.append({
            "qid": f"spider_dev_{idx}",
            "original_id": idx,
            "dataset": "spider",
            "db_id": raw["db_id"],
            "difficulty": hardness,
            "question": raw["question"],
            "gold_sql": raw["query"],
            "evidence": "",
        })

    spider_dev_200, spider_remaining = stratified_split(spider_items, n_sample=200, seed=seed)
    spider_tune_50, _ = stratified_split(spider_remaining, n_sample=50, seed=seed)

    # 2. Process BIRD Mini-Dev
    bird_path = Path("./data/bird/mini_dev_sqlite.json")
    if not bird_path.exists():
        raise FileNotFoundError(f"Missing BIRD mini-dev data at {bird_path}")

    with open(bird_path, "r", encoding="utf-8") as f:
        bird_raw = json.load(f)

    bird_items = []
    for raw in bird_raw:
        bird_items.append({
            "qid": f"bird_mini_{raw['question_id']}",
            "original_id": raw["question_id"],
            "dataset": "bird",
            "db_id": raw["db_id"],
            "difficulty": raw.get("difficulty", "moderate").lower(),
            "question": raw["question"],
            "gold_sql": raw["SQL"],
            "evidence": raw.get("evidence", ""),
        })

    bird_mini_200, bird_remaining = stratified_split(bird_items, n_sample=200, seed=seed)
    bird_tune_50, _ = stratified_split(bird_remaining, n_sample=50, seed=seed)

    # Combined tune 50 (25 from Spider, 25 from BIRD, strictly disjoint from eval 200s)
    sp_tune_25, _ = stratified_split(spider_remaining, n_sample=25, seed=seed)
    bd_tune_25, _ = stratified_split(bird_remaining, n_sample=25, seed=seed)
    combined_tune_50 = sp_tune_25 + bd_tune_25

    # 3. Catalog DB list (20 Spider Dev DBs)
    catalog_dbs = sorted(list({item["db_id"] for item in spider_items}))

    # Save to disk
    subsets_map = {
        "spider_dev_200.json": spider_dev_200,
        "spider_tune_50.json": spider_tune_50,
        "bird_mini_200.json": bird_mini_200,
        "bird_tune_50.json": bird_tune_50,
        "tune_50.json": combined_tune_50,
    }

    for fname, data in subsets_map.items():
        file_path = out_dir / fname
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s with %d questions", fname, len(data))

    catalog_path = out_dir / "catalog_dbs.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog_dbs, f, indent=2)
    logger.info("Saved catalog_dbs.json with %d databases: %s", len(catalog_dbs), catalog_dbs)

    # Print distributions
    print("\n" + "=" * 65)
    print("SUBSET DISTRIBUTIONS (Seed 42)")
    print("=" * 65)

    def print_dist(title: str, items: List[Dict[str, Any]]):
        diff_cnt = collections.Counter(it["difficulty"] for it in items)
        db_cnt = collections.Counter(it["db_id"] for it in items)
        print(f"\n--- {title} (Total: {len(items)}) ---")
        print("Difficulty distribution:")
        for d, count in sorted(diff_cnt.items()):
            print(f"  {d:12s}: {count:3d} ({count / len(items) * 100:.1f}%)")
        print(f"Unique databases: {len(db_cnt)}")

    print_dist("Spider Dev Full (1034)", spider_items)
    print_dist("Spider Dev 200 (eval)", spider_dev_200)
    print_dist("Spider Tune 50 (tune)", spider_tune_50)
    print_dist("BIRD Mini-Dev Full (500)", bird_items)
    print_dist("BIRD Mini 200 (eval)", bird_mini_200)
    print_dist("BIRD Tune 50 (tune)", bird_tune_50)
    print_dist("Combined Tune 50", combined_tune_50)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    build_subsets()

