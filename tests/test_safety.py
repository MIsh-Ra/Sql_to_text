"""Tests for SQLite sandbox and AST SQL validator security controls."""

import sqlite3
from pathlib import Path

import pytest

from app.db import QuerySecurityError, QueryTimeoutError, execute_query
from app.validate import validate


@pytest.fixture
def sample_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with a users and orders table."""
    db_file = tmp_path / "test_safety.sqlite"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);")
    cursor.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);")
    cursor.executemany(
        "INSERT INTO users VALUES (?, ?, ?);",
        [(1, "Alice", "alice@example.com"), (2, "Bob", "bob@example.com"), (3, "Charlie", "charlie@example.com")],
    )
    cursor.executemany(
        "INSERT INTO orders VALUES (?, ?, ?);",
        [(101, 1, 99.5), (102, 1, 20.0), (103, 2, 150.0)],
    )
    conn.commit()
    conn.close()
    return db_file


# --- AST Validator Tests ---

def test_validator_allowed_queries():
    schema = {"users": ["id", "name", "email"], "orders": ["id", "user_id", "amount"]}
    # Simple select
    res = validate("SELECT name FROM users WHERE id = 1;", schema)
    assert res.ok, res.message

    # Join
    res = validate("SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id;", schema)
    assert res.ok, res.message

    # CTE
    res = validate("WITH high_orders AS (SELECT user_id, amount FROM orders WHERE amount > 50) SELECT * FROM high_orders;", schema)
    assert res.ok, res.message

    # Union
    res = validate("SELECT name FROM users UNION SELECT name FROM users;", schema)
    assert res.ok, res.message


def test_validator_blocks_stacked_statements():
    res = validate("SELECT * FROM users; DROP TABLE users;")
    assert not res.ok
    assert res.error_type == "STACKED_STATEMENTS"


def test_validator_blocks_comment_tricks():
    # Comment hiding a second statement
    res = validate("SELECT * FROM users -- comment\n; DROP TABLE users;")
    assert not res.ok
    assert res.error_type in ["STACKED_STATEMENTS", "DISALLOWED_NODE", "PARSE_ERROR"]


def test_validator_blocks_ddl_and_dml():
    # DROP
    assert not validate("DROP TABLE users;").ok
    # INSERT
    assert not validate("INSERT INTO users VALUES (4, 'Eve', 'eve@example.com');").ok
    # UPDATE
    assert not validate("UPDATE users SET name = 'Hacked';").ok
    # DELETE
    assert not validate("DELETE FROM users;").ok
    # ALTER
    assert not validate("ALTER TABLE users ADD COLUMN age INT;").ok


def test_validator_blocks_pragma_and_attach():
    assert not validate("PRAGMA table_info(users);").ok
    assert not validate("ATTACH DATABASE ':memory:' AS rogue;").ok


def test_validator_blocks_load_extension():
    res = validate("SELECT load_extension('evil.dll');")
    assert not res.ok
    assert res.error_type in ["DANGEROUS_FUNCTION", "DISALLOWED_NODE"]


def test_validator_unknown_tables_and_columns():
    schema = {"users": ["id", "name", "email"]}
    # Unknown table
    res_tbl = validate("SELECT * FROM credit_cards;", schema)
    assert not res_tbl.ok
    assert res_tbl.error_type == "UNKNOWN_TABLE"

    # Unknown column
    res_col = validate("SELECT password_hash FROM users;", schema)
    assert not res_col.ok
    assert res_col.error_type == "UNKNOWN_COLUMN"


# --- Sandbox Execution Tests ---

def test_sandbox_allowed_select(sample_db: Path):
    res = execute_query(sample_db, "SELECT name, email FROM users ORDER BY id ASC;")
    assert res.columns == ["name", "email"]
    assert res.row_count == 3
    assert res.rows[0] == ("Alice", "alice@example.com")
    assert not res.truncated


def test_sandbox_blocks_writes(sample_db: Path):
    with pytest.raises(QuerySecurityError):
        execute_query(sample_db, "INSERT INTO users VALUES (4, 'Eve', 'eve@test.com');")

    with pytest.raises(QuerySecurityError):
        execute_query(sample_db, "DROP TABLE users;")

    with pytest.raises(QuerySecurityError):
        execute_query(sample_db, "CREATE TABLE evil (id INT);")


def test_sandbox_blocks_attach_and_pragma(sample_db: Path):
    with pytest.raises(QuerySecurityError):
        execute_query(sample_db, "ATTACH DATABASE ':memory:' AS rogue;")

    with pytest.raises(QuerySecurityError):
        execute_query(sample_db, "PRAGMA journal_mode=WAL;")


def test_sandbox_row_cap(sample_db: Path):
    # Cap to 2 rows
    res = execute_query(sample_db, "SELECT * FROM users;", max_rows=2)
    assert res.row_count == 2
    assert res.truncated is True

    # No cap
    res_all = execute_query(sample_db, "SELECT * FROM users;", max_rows=None)
    assert res_all.row_count == 3
    assert res_all.truncated is False


def test_sandbox_timeout(sample_db: Path):
    # Recursive CTE creating an expensive loop to trigger timeout
    infinite_query = """
    WITH RECURSIVE cnt(x) AS (
        SELECT 1
        UNION ALL
        SELECT x + 1 FROM cnt WHERE x < 100000000
    )
    SELECT count(*) FROM cnt;
    """
    with pytest.raises(QueryTimeoutError):
        execute_query(sample_db, infinite_query, timeout_s=0.2)

