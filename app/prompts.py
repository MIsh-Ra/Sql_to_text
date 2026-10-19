"""All prompt templates for generation, repair, and summarization."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

SYSTEM_GENERATE = (
    "You are an expert SQLite analyst. Write exactly one SQLite SELECT query that answers "
    "the question. Use only tables and columns that appear in the schema, spelled exactly. "
    "Prefer simple, correct queries. Return only the SQL inside one sql code block and nothing else."
)


def format_generate_prompt(
    schema_text: str,
    question: str,
    evidence: Optional[str] = None,
    fewshots: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Format user prompt for SQL generation."""
    sections = []

    # 1. Schema
    sections.append(f"### Schema\n{schema_text.strip()}")

    # 2. Examples (only when provided)
    if fewshots:
        examples_text = []
        for i, ex in enumerate(fewshots, 1):
            examples_text.append(f"Example {i}:\nQuestion: {ex['question']}\nSQL:\n```sql\n{ex['sql'].strip()}\n```")
        sections.append("### Examples\n" + "\n\n".join(examples_text))

    # 3. Hint (only when evidence exists)
    if evidence and evidence.strip():
        sections.append(f"### Hint\n{evidence.strip()}")

    # 4. Question
    sections.append(f"### Question\n{question.strip()}")

    return "\n\n".join(sections)


def format_repair_prompt(
    schema_text: str,
    question: str,
    previous_sql: str,
    error_message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format user prompt for SQL repair."""
    sections = []

    sections.append(f"### Schema\n{schema_text.strip()}")
    sections.append(f"### Question\n{question.strip()}")
    sections.append(f"### Failing SQL\n```sql\n{previous_sql.strip()}\n```")
    sections.append(f"### Error Message\n{error_message.strip()}")

    if history and len(history) > 1:
        history_text = []
        for i, item in enumerate(history[:-1], 1):
            history_text.append(
                f"Attempt {i}:\nSQL: {item.get('sql', '').strip()}\nError: {item.get('error', '').strip()}"
            )
        sections.append("### Earlier Failed Attempts\n" + "\n\n".join(history_text))

    instructions = (
        "Fix the query so it answers the question. Do not repeat a failed query. "
        "Return only the corrected SQL in one sql code block."
    )
    sections.append(instructions)

    return "\n\n".join(sections)


def format_summarize_prompt(
    question: str,
    sql: str,
    rows: List[Any],
    col_names: Optional[List[str]] = None,
) -> str:
    """Format prompt for concise summary of query results."""
    # Truncate to first 20 rows
    preview_rows = rows[:20] if rows else []

    sections = [
        f"Question: {question.strip()}",
        f"SQL: {sql.strip()}",
    ]

    if not preview_rows:
        sections.append("Query Result: Empty (0 rows returned)")
    else:
        header = f"Columns: {', '.join(col_names)}" if col_names else "Rows:"
        rows_str = "\n".join(str(r) for r in preview_rows)
        sections.append(f"{header}\n{rows_str}")

    sections.append(
        "Instructions: Answer the question in at most two sentences using only values present in the rows. "
        "If the result is empty, say so. Do not invent or assume any values absent from the rows."
    )

    return "\n\n".join(sections)


def get_prompts_hash() -> str:
    """Compute deterministic SHA256 hash of this prompts file."""
    path = Path(__file__).resolve()
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

