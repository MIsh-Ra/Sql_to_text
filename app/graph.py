"""LangGraph state machine for the Text-to-SQL Agent with repair loop and checkpointing."""

import logging
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.db import (
    execute_query,
    full_schema,
    get_db_path,
    get_table_schemas,
)
from app.llm import get_llm
from app.prompts import (
    SYSTEM_GENERATE,
    format_generate_prompt,
    format_repair_prompt,
    format_summarize_prompt,
)
from app.state import AgentConfig, AgentState
from app.validate import validate

logger = logging.getLogger(__name__)


def extract_sql_from_response(text: str) -> str:
    """Extract clean SQL string from LLM output, handling markdown blocks or raw queries."""
    # Look for ```sql ... ```
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
    else:
        # Fallback to entire text
        extracted = text.strip()

    # Strip trailing semicolons
    return extracted.rstrip(";").strip()


# --- Node Definitions ---

def route_db_node(state: AgentState) -> Dict[str, Any]:
    """Catalog mode router: selects db_id from question when not provided."""
    # If already set, pass through
    if state.get("db_id"):
        return {}

    # In Phase 3, we hook up multi-db catalog retrieval
    # For now, if catalog_mode is on and schema_index is available, route_catalog is used
    try:
        from app.schema_index import route_catalog_db
        db_id = route_catalog_db(state["question"])
        return {"db_id": db_id}
    except Exception:
        # Default fallback
        return {"db_id": "concert_singer"}


def retrieve_schema_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve database schema, either via compact retrieval or full DDL."""
    config = state.get("config", {})
    db_id = state.get("db_id")
    if not db_id:
        return {"schema_text": "", "tables": [], "columns": [], "schema_truncated": False}

    use_retrieval = config.get("use_retrieval", False)
    if use_retrieval:
        try:
            from app.schema_index import retrieve_schema
            k_tbl = config.get("k_tables", 6)
            k_col = config.get("k_columns", 25)
            hybrid = config.get("hybrid", False)
            schema_text, tables, columns, trunc = retrieve_schema(
                question=state["question"],
                db_id=db_id,
                k_tables=k_tbl,
                k_columns=k_col,
                hybrid=hybrid,
            )
            return {
                "schema_text": schema_text,
                "tables": tables,
                "columns": columns,
                "schema_truncated": trunc,
            }
        except Exception as exc:
            logger.debug("Retrieval failed, falling back to full schema: %s", exc)

    # Baseline fallback: full schema
    schema_text, tables, columns, trunc = full_schema(db_id)
    return {
        "schema_text": schema_text,
        "tables": tables,
        "columns": columns,
        "schema_truncated": trunc,
    }


def retrieve_fewshots_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve few-shot demonstrations if enabled."""
    config = state.get("config", {})
    if not config.get("use_fewshot", False):
        return {"fewshots": []}

    try:
        from app.fewshot_index import retrieve_fewshots
        k = config.get("k_fewshots", 3)
        fewshots = retrieve_fewshots(
            question=state["question"],
            db_id=state.get("db_id", ""),
            k=k,
        )
        return {"fewshots": fewshots}
    except Exception as exc:
        logger.debug("Fewshot retrieval failed: %s", exc)
        return {"fewshots": []}


def generate_sql_node(state: AgentState) -> Dict[str, Any]:
    """Generate SQL from question, schema, and optional few-shots."""
    t0 = time.perf_counter()
    llm = get_llm()

    user_prompt = format_generate_prompt(
        schema_text=state.get("schema_text", ""),
        question=state["question"],
        evidence=state.get("evidence"),
        fewshots=state.get("fewshots"),
    )

    messages = [
        SystemMessage(content=SYSTEM_GENERATE),
        HumanMessage(content=user_prompt),
    ]

    res = llm.invoke(messages)
    sql = extract_sql_from_response(res.content)
    latency = time.perf_counter() - t0

    usage = getattr(res, "usage_metadata", {}) or {}
    metrics = dict(state.get("metrics") or {})
    metrics["tokens_in"] = metrics.get("tokens_in", 0) + usage.get("input_tokens", 0)
    metrics["tokens_out"] = metrics.get("tokens_out", 0) + usage.get("output_tokens", 0)
    metrics["llm_calls"] = metrics.get("llm_calls", 0) + 1
    pnl = dict(metrics.get("per_node_latency") or {})
    pnl["generate_sql"] = pnl.get("generate_sql", 0.0) + latency
    metrics["per_node_latency"] = pnl

    return {
        "sql": sql,
        "attempts": 1,
        "history": [],
        "metrics": metrics,
    }


def validate_node(state: AgentState) -> Dict[str, Any]:
    """Run AST validation on current SQL against database schema."""
    sql = state.get("sql", "")
    db_id = state.get("db_id")

    schema_dict = None
    if db_id:
        try:
            db_path = get_db_path(db_id)
            schema_dict = get_table_schemas(db_path)
        except Exception:
            schema_dict = None

    v_res = validate(sql, schema=schema_dict)
    if not v_res.ok:
        err = f"[{v_res.error_type}] {v_res.message}"
        return {"validation_error": err}
    return {"validation_error": None}


def execute_node(state: AgentState) -> Dict[str, Any]:
    """Execute SQL query inside SQLite sandbox."""
    sql = state.get("sql", "")
    db_id = state.get("db_id")
    if not db_id:
        return {"exec_error": "No database specified for execution."}

    try:
        db_path = get_db_path(db_id)
        # In eval or when max_returned_rows is None, fetch unlimited
        res = execute_query(db_path, sql, max_rows=settings.max_returned_rows)
        return {
            "rows": res.rows,
            "col_names": res.columns,
            "row_count": res.row_count,
            "truncated": res.truncated,
            "exec_error": None,
        }
    except Exception as exc:
        return {"exec_error": str(exc), "rows": None}


def repair_node(state: AgentState) -> Dict[str, Any]:
    """Attempt to repair a failing SQL query with error feedback."""
    t0 = time.perf_counter()
    llm = get_llm()

    current_sql = state.get("sql", "")
    err_msg = state.get("validation_error") or state.get("exec_error") or "Unknown error"
    history = list(state.get("history") or [])
    history.append({"sql": current_sql, "error": err_msg})

    attempts = state.get("attempts", 1) + 1

    prompt = format_repair_prompt(
        schema_text=state.get("schema_text", ""),
        question=state["question"],
        previous_sql=current_sql,
        error_message=err_msg,
        history=history,
    )

    messages = [
        SystemMessage(content=SYSTEM_GENERATE),
        HumanMessage(content=prompt),
    ]

    res = llm.invoke(messages)
    new_sql = extract_sql_from_response(res.content)
    latency = time.perf_counter() - t0

    usage = getattr(res, "usage_metadata", {}) or {}
    metrics = dict(state.get("metrics") or {})
    metrics["tokens_in"] = metrics.get("tokens_in", 0) + usage.get("input_tokens", 0)
    metrics["tokens_out"] = metrics.get("tokens_out", 0) + usage.get("output_tokens", 0)
    metrics["llm_calls"] = metrics.get("llm_calls", 0) + 1
    pnl = dict(metrics.get("per_node_latency") or {})
    pnl["repair"] = pnl.get("repair", 0.0) + latency
    metrics["per_node_latency"] = pnl

    return {
        "sql": new_sql,
        "attempts": attempts,
        "history": history,
        "validation_error": None,
        "exec_error": None,
        "metrics": metrics,
    }


def summarize_node(state: AgentState) -> Dict[str, Any]:
    """Generate concise human answer from query result."""
    rows = state.get("rows") or []
    col_names = state.get("col_names") or []
    question = state.get("question", "")
    sql = state.get("sql", "")

    # For empty results or simple queries, return direct summary
    if not rows:
        return {"answer": "The query executed successfully but returned 0 rows.", "status": "ok"}

    try:
        prompt = format_summarize_prompt(question, sql, rows, col_names)
        llm = get_llm()
        res = llm.invoke([HumanMessage(content=prompt)])
        answer = res.content.strip()
    except Exception:
        answer = f"Found {len(rows)} matching record(s)."

    return {"answer": answer, "status": "ok"}


def fail_node(state: AgentState) -> Dict[str, Any]:
    """Terminal node for execution or validation failures."""
    last_err = state.get("validation_error") or state.get("exec_error") or "Execution failed."
    return {
        "status": "failed",
        "answer": f"Failed to generate valid SQL. Last error: {last_err}",
    }


# --- Conditional Routing Edges ---

def route_after_validate(state: AgentState) -> str:
    """Route after validate: ok -> execute, error -> repair or fail."""
    if not state.get("validation_error"):
        return "execute"

    config = state.get("config", {})
    max_att = config.get("max_attempts", 3)
    use_repair = config.get("use_repair", False)

    if use_repair and state.get("attempts", 1) < max_att:
        return "repair"
    return "fail"


def route_after_execute(state: AgentState) -> str:
    """Route after execute: ok -> summarize, error -> repair or fail."""
    if not state.get("exec_error"):
        return "summarize"

    config = state.get("config", {})
    max_att = config.get("max_attempts", 3)
    use_repair = config.get("use_repair", False)

    if use_repair and state.get("attempts", 1) < max_att:
        return "repair"
    return "fail"


def route_after_repair(state: AgentState) -> str:
    """Loop guard check: if repaired SQL is identical to any earlier attempt, fail immediately."""
    new_sql = state.get("sql", "").strip()
    history = state.get("history") or []

    for item in history:
        if item.get("sql", "").strip() == new_sql:
            logger.info("Loop guard triggered: identical SQL generated during repair.")
            return "fail"

    return "validate"


# --- Graph Construction ---

def build_graph(checkpointer: Optional[Any] = None) -> Any:
    """Construct and compile the LangGraph workflow state machine."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("route_db", route_db_node)
    workflow.add_node("retrieve_schema", retrieve_schema_node)
    workflow.add_node("retrieve_fewshots", retrieve_fewshots_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("fail", fail_node)

    # Edges
    workflow.add_edge(START, "route_db")
    workflow.add_edge("route_db", "retrieve_schema")
    workflow.add_edge("retrieve_schema", "retrieve_fewshots")
    workflow.add_edge("retrieve_fewshots", "generate_sql")
    workflow.add_edge("generate_sql", "validate")

    workflow.add_conditional_edges(
        "validate",
        route_after_validate,
        {"execute": "execute", "repair": "repair", "fail": "fail"},
    )

    workflow.add_conditional_edges(
        "execute",
        route_after_execute,
        {"summarize": "summarize", "repair": "repair", "fail": "fail"},
    )

    workflow.add_conditional_edges(
        "repair",
        route_after_repair,
        {"validate": "validate", "fail": "fail"},
    )

    workflow.add_edge("summarize", END)
    workflow.add_edge("fail", END)

    return workflow.compile(checkpointer=checkpointer)


# Singleton checkpointer
_CHECKPOINT_SAVER: Optional[SqliteSaver] = None


def get_checkpointer() -> SqliteSaver:
    global _CHECKPOINT_SAVER
    if _CHECKPOINT_SAVER is None:
        db_path = Path("./data/checkpoints.sqlite")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _CHECKPOINT_SAVER = SqliteSaver(conn)
    return _CHECKPOINT_SAVER


def run(
    question: str,
    db_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    thread_id: Optional[str] = None,
    evidence: Optional[str] = None,
    checkpointer: Optional[Any] = None,
) -> AgentState:
    """Run agent synchronously to completion."""
    cfg: AgentConfig = config or {
        "use_retrieval": False,
        "hybrid": False,
        "use_fewshot": False,
        "use_repair": False,
        "max_attempts": 1,
    }

    saver = checkpointer or get_checkpointer()
    graph = build_graph(checkpointer=saver)

    session_thread = thread_id or str(uuid.uuid4())
    initial_state: AgentState = {
        "question": question,
        "db_id": db_id,
        "evidence": evidence,
        "config": cfg,
        "attempts": 0,
        "history": [],
        "metrics": {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0, "per_node_latency": {}},
    }

    t0 = time.perf_counter()
    res = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": session_thread}},
    )
    total_lat = time.perf_counter() - t0
    res_metrics = dict(res.get("metrics") or {})
    res_metrics["total_latency_s"] = total_lat
    res["metrics"] = res_metrics
    return res


def stream(
    question: str,
    db_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    thread_id: Optional[str] = None,
    evidence: Optional[str] = None,
    checkpointer: Optional[Any] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream per-node state updates."""
    cfg: AgentConfig = config or {
        "use_retrieval": False,
        "hybrid": False,
        "use_fewshot": False,
        "use_repair": False,
        "max_attempts": 1,
    }

    saver = checkpointer or get_checkpointer()
    graph = build_graph(checkpointer=saver)
    session_thread = thread_id or str(uuid.uuid4())

    initial_state: AgentState = {
        "question": question,
        "db_id": db_id,
        "evidence": evidence,
        "config": cfg,
        "attempts": 0,
        "history": [],
        "metrics": {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0, "per_node_latency": {}},
    }

    for update in graph.stream(
        initial_state,
        config={"configurable": {"thread_id": session_thread}},
        stream_mode="updates",
    ):
        yield update
