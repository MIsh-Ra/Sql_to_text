"""Unit tests for evaluation metrics, edge cases, bootstrap CI, and McNemar test."""

from eval.metrics import (
    bootstrap_ci,
    exact_mcnemar_test,
    execution_match,
    extract_gold_tables_and_columns,
)


def test_execution_match_float_rounding_and_nulls():
    gold = [(1, 3.14159265, None, "Alice")]
    pred = [(1, 3.14158, None, "Alice ")]  # Rounds to 3.1416, whitespace stripped

    set_match, multi_match = execution_match(pred, gold)
    assert set_match is True
    assert multi_match is True


def test_execution_match_ordering():
    # Order should not matter for set or multiset match
    gold = [("Alice", 10), ("Bob", 20)]
    pred = [("Bob", 20), ("Alice", 10)]

    set_match, multi_match = execution_match(pred, gold)
    assert set_match is True
    assert multi_match is True


def test_execution_match_duplicates_set_vs_multiset():
    # Duplicates match in set match, but fail in multiset match
    gold = [("Alice", 10)]
    pred = [("Alice", 10), ("Alice", 10)]

    set_match, multi_match = execution_match(pred, gold)
    assert set_match is True
    assert multi_match is False


def test_bootstrap_ci():
    # Array of all 1s should have mean 1.0 and CI [1.0, 1.0]
    m, lo, hi = bootstrap_ci([1.0] * 100)
    assert m == 1.0
    assert lo == 1.0
    assert hi == 1.0

    # 50% ones, 50% zeros
    m, lo, hi = bootstrap_ci([1.0] * 50 + [0.0] * 50)
    assert m == 0.5
    assert 0.35 <= lo <= 0.45
    assert 0.55 <= hi <= 0.65


def test_mcnemar_test_known():
    # A = [1, 1, 0, 0], B = [1, 0, 1, 0]
    # b: A=0, B=1 -> 1
    # c: A=1, B=0 -> 1
    # Equal b and c -> p-value = 1.0
    a = [True, True, False, False]
    b = [True, False, True, False]
    b_count, c_count, p_val = exact_mcnemar_test(a, b)
    assert b_count == 1
    assert c_count == 1
    assert p_val == 1.0

    # A dominates B completely
    a_dom = [True] * 20
    b_weak = [False] * 20
    b_c, c_c, p_dom = exact_mcnemar_test(a_dom, b_weak)
    assert b_c == 0
    assert c_c == 20
    assert p_dom < 0.001


def test_gold_sql_table_and_column_extraction():
    sql = "SELECT u.name, avg(o.amount) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name;"
    tables, cols = extract_gold_tables_and_columns(sql)
    assert "users" in tables
    assert "orders" in tables
    assert "name" in cols
    assert "amount" in cols

