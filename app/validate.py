"""AST-based SQL validator using sqlglot for SQLite dialect."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer import qualify

logger = logging.getLogger(__name__)

DISALLOWED_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
    exp.Pragma,
    exp.Attach,
    exp.TruncateTable,
)

DISALLOWED_FUNCTIONS = {
    "load_extension",
    "writefile",
    "readfile",
    "edit",
}


@dataclass
class ValidationResult:
    """Structured validation outcome."""
    ok: bool
    error_type: Optional[str] = None
    message: Optional[str] = None
    ast: Optional[exp.Expression] = None


def _is_allowed_query_root(node: exp.Expression) -> bool:
    """Root must be SELECT, a set operation (UNION/INTERSECT/EXCEPT), or a WITH ending in SELECT."""
    if isinstance(node, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return True
    if isinstance(node, exp.With):
        this = node.this
        return isinstance(this, (exp.Select, exp.Union, exp.Intersect, exp.Except))
    return False


def validate(
    sql: str,
    schema: Optional[Union[Dict[str, List[str]], Dict[str, Dict[str, str]]]] = None,
) -> ValidationResult:
    """Validate that SQL is a single, safe, read-only SELECT conforming to schema.

    Args:
        sql: SQL query string.
        schema: Optional schema mapping table names to column lists or column dicts.

    Returns:
        ValidationResult with ok flag, error_type, and descriptive message.
    """
    clean_sql = sql.strip().rstrip(";")
    if not clean_sql:
        return ValidationResult(ok=False, error_type="EMPTY_SQL", message="SQL statement is empty.")

    # 1. Parse statement(s)
    try:
        statements = sqlglot.parse(clean_sql, read="sqlite")
    except ParseError as exc:
        return ValidationResult(ok=False, error_type="SYNTAX_ERROR", message=f"Syntax error: {exc}")
    except Exception as exc:
        return ValidationResult(ok=False, error_type="PARSE_ERROR", message=f"Failed to parse SQL: {exc}")

    # Reject empty or stacked statements
    statements = [s for s in statements if s is not None]
    if len(statements) == 0:
        return ValidationResult(ok=False, error_type="EMPTY_SQL", message="No valid SQL statement found.")
    if len(statements) > 1:
        return ValidationResult(
            ok=False,
            error_type="STACKED_STATEMENTS",
            message="Multiple SQL statements are strictly prohibited.",
        )

    root = statements[0]

    # 2. Check root statement type
    if not _is_allowed_query_root(root):
        return ValidationResult(
            ok=False,
            error_type="DISALLOWED_STATEMENT",
            message=f"Root statement must be a SELECT or CTE. Found: {root.__class__.__name__}",
        )

    # 3. Walk AST for any disallowed write / DDL nodes
    for node in root.walk():
        if isinstance(node, DISALLOWED_NODES):
            return ValidationResult(
                ok=False,
                error_type="DISALLOWED_NODE",
                message=f"Disallowed operation detected: {node.__class__.__name__}",
            )
        # Block dangerous functions
        if isinstance(node, (exp.Func, exp.Anonymous)):
            func_name = (node.name or "").lower()
            if func_name in DISALLOWED_FUNCTIONS:
                return ValidationResult(
                    ok=False,
                    error_type="DANGEROUS_FUNCTION",
                    message=f"Function '{func_name}' is prohibited.",
                )

    # 4. Check CTE bodies if present
    for cte in getattr(root, "ctes", []):
        cte_this = cte.this
        if not isinstance(cte_this, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            return ValidationResult(
                ok=False,
                error_type="DISALLOWED_STATEMENT",
                message="CTE definition body must be a SELECT expression.",
            )

    # 5. Schema verification if schema is provided
    if schema:
        # Normalize schema: dict of table_name.lower() -> dict of col_name.lower() -> "TEXT"
        normalized_schema: Dict[str, Dict[str, str]] = {}
        for tbl, cols in schema.items():
            tbl_lower = tbl.lower()
            if isinstance(cols, dict):
                normalized_schema[tbl_lower] = {c.lower(): str(t) for c, t in cols.items()}
            elif isinstance(cols, list):
                normalized_schema[tbl_lower] = {c.lower(): "TEXT" for c in cols}

        # Collect CTE names defined in query
        cte_names: Set[str] = set()
        for cte in getattr(root, "ctes", []):
            if cte.alias:
                cte_names.add(cte.alias.lower())

        # Check tables
        for table in root.find_all(exp.Table):
            tbl_name = table.name.lower()
            if not tbl_name:
                continue
            if tbl_name not in normalized_schema and tbl_name not in cte_names:
                return ValidationResult(
                    ok=False,
                    error_type="UNKNOWN_TABLE",
                    message=f"Table '{table.name}' does not exist in schema.",
                )

        # Check columns with sqlglot qualification
        try:
            # We qualify on a copy to avoid mutating the AST
            root_copy = root.copy()
            qualify.qualify(
                root_copy,
                schema=normalized_schema,
                dialect="sqlite",
                validate_qualify_columns=True,
            )
        except OptimizeError as exc:
            msg = str(exc)
            if "could not be resolved" in msg.lower() or "unknown column" in msg.lower():
                return ValidationResult(
                    ok=False,
                    error_type="UNKNOWN_COLUMN",
                    message=f"Column resolution error: {exc}",
                )
            # Other optimization issues: log and allow SQLite engine to handle
            logger.debug("SQLGlot qualify optimization skipped: %s", exc)
        except Exception as exc:
            logger.debug("SQLGlot column qualification skipped due to exception: %s", exc)

    return ValidationResult(ok=True, ast=root)
