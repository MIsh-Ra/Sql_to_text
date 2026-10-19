"""Safe SQLite execution sandbox with read-only authorizer, progress handler timeout, and row limits."""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.config import settings

# Allowed SQLite authorizer actions for read-only analytical queries
ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),  # 33 is SQLITE_RECURSIVE
}


def _sqlite_authorizer(action_code: int, arg1: Any, arg2: Any, db_name: Any, trigger_name: Any) -> int:
    """Strict authorizer: allow ONLY SELECT, READ, FUNCTION, and RECURSIVE; deny all others."""
    if action_code in ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class QueryTimeoutError(Exception):
    """Raised when a query exceeds the configured execution timeout."""
    pass


class QuerySecurityError(Exception):
    """Raised when a query violates sandbox security constraints."""
    pass


@dataclass
class QueryResult:
    """Structured result of executing a SQL query in the sandbox."""
    columns: List[str]
    rows: List[Tuple[Any, ...]]
    row_count: int
    truncated: bool
    execution_time_s: float


def execute_query(
    db_path: Union[str, Path],
    sql: str,
    timeout_s: Optional[float] = None,
    max_rows: Optional[int] = None,
) -> QueryResult:
    """Execute a SQL query inside a secured read-only SQLite sandbox.

    Args:
        db_path: Absolute or relative path to the SQLite database file.
        sql: The SQL query to execute.
        timeout_s: Query execution timeout in seconds (defaults to settings.query_timeout_s).
        max_rows: Maximum rows to return (None for unlimited in eval, or int cap for UI/API).

    Returns:
        QueryResult with column names, tuples of values, and execution metadata.
    """
    resolved_path = Path(db_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Database file does not exist: {resolved_path}")

    timeout = timeout_s if timeout_s is not None else settings.query_timeout_s
    # SQLite URI with read-only mode
    # For Windows paths, normalize slashes
    uri_path = resolved_path.as_posix()
    db_uri = f"file:///{uri_path}?mode=ro" if not uri_path.startswith("/") else f"file://{uri_path}?mode=ro"

    conn = sqlite3.connect(db_uri, uri=True, timeout=1.0)
    try:
        # Install authorizer
        conn.set_authorizer(_sqlite_authorizer)

        # Install progress handler for query timeout
        start_time = time.perf_counter()
        deadline = start_time + timeout

        def progress_handler() -> int:
            if time.perf_counter() > deadline:
                return 1  # Non-zero aborts the query
            return 0

        # Check timeout every 500 virtual machine instructions
        conn.set_progress_handler(progress_handler, 500)

        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            col_names = [desc[0] for desc in cursor.description] if cursor.description else []

            truncated = False
            if max_rows is not None:
                # Fetch up to max_rows + 1 to detect truncation without auto-injecting LIMIT
                fetched = cursor.fetchmany(max_rows + 1)
                if len(fetched) > max_rows:
                    truncated = True
                    rows = fetched[:max_rows]
                else:
                    rows = fetched
            else:
                rows = cursor.fetchall()

            exec_time = time.perf_counter() - start_time
            return QueryResult(
                columns=col_names,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                execution_time_s=exec_time,
            )
        except sqlite3.OperationalError as exc:
            err_msg = str(exc).lower()
            if "interrupted" in err_msg or "abort" in err_msg:
                raise QueryTimeoutError(f"Query timed out after {timeout:.1f}s") from exc
            if "not authorized" in err_msg:
                raise QuerySecurityError(f"Security sandbox denied query action: {exc}") from exc
            raise
        except sqlite3.DatabaseError as exc:
            if "not authorized" in str(exc).lower():
                raise QuerySecurityError(f"Security sandbox denied query action: {exc}") from exc
            raise
    finally:
        conn.close()



# Schema extraction and database resolution helpers

def get_db_path(db_id: str) -> Path:
    """Resolve database path across Spider, BIRD, or demo locations."""
    candidates = [
        Path(f"./data/spider/database/{db_id}/{db_id}.sqlite"),
        Path(f"./data/bird/dev_databases/{db_id}/{db_id}.sqlite"),
        Path(f"./demo_data/{db_id}.sqlite"),
        Path(f"./demo_data/{db_id}/{db_id}.sqlite"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    # Search in data/
    found = list(Path("./data").glob(f"**/{db_id}.sqlite"))
    if found:
        return found[0].resolve()

    # Search in demo_data/
    found_demo = list(Path("./demo_data").glob(f"**/{db_id}.sqlite"))
    if found_demo:
        return found_demo[0].resolve()

    raise FileNotFoundError(f"Database '{db_id}.sqlite' not found in data or demo_data directories.")


def get_table_schemas(db_path: Path) -> Dict[str, List[str]]:
    """Return dictionary of table_name -> [col_name1, col_name2, ...]."""
    uri_path = db_path.resolve().as_posix()
    db_uri = f"file:///{uri_path}?mode=ro" if not uri_path.startswith("/") else f"file://{uri_path}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True, timeout=2.0)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cursor.fetchall()]
        result = {}
        for t in tables:
            cursor.execute(f"PRAGMA table_info(\"{t}\");")
            cols = [r[1] for r in cursor.fetchall()]
            result[t] = cols
        return result
    finally:
        conn.close()


def full_schema(
    db_id: str,
    max_chars: int = 12000,
) -> Tuple[str, List[str], List[str], bool]:
    """Render full DDL schema for a database, truncated to char/token budget with a marker.

    Returns:
        (schema_text, table_names, column_names, is_truncated)
    """
    db_path = get_db_path(db_id)
    uri_path = db_path.resolve().as_posix()
    db_uri = f"file:///{uri_path}?mode=ro" if not uri_path.startswith("/") else f"file://{uri_path}?mode=ro"

    conn = sqlite3.connect(db_uri, uri=True, timeout=2.0)
    tables: List[str] = []
    columns: List[str] = []
    ddl_statements: List[str] = []

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        for tbl_name, sql in cursor.fetchall():
            tables.append(tbl_name)
            if sql:
                ddl_statements.append(sql.strip() + ";")
            # Collect columns
            cursor.execute(f"PRAGMA table_info(\"{tbl_name}\");")
            for col_row in cursor.fetchall():
                columns.append(col_row[1])
    finally:
        conn.close()

    full_text = "\n\n".join(ddl_statements)
    is_truncated = False
    if len(full_text) > max_chars:
        is_truncated = True
        full_text = full_text[:max_chars] + "\n\n/* [SCHEMA TRUNCATED DUE TO TOKEN BUDGET] */"

    return full_text, tables, columns, is_truncated
