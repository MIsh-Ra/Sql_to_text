"""LangGraph State definition for the Text-to-SQL Agent."""

from typing import Any, Dict, List, Optional, TypedDict


class AgentConfig(TypedDict, total=False):
    """Feature flags and execution parameters controlling the agent's behavior."""
    use_retrieval: bool  # False for C0 baseline (uses full_schema)
    hybrid: bool         # Dense + BM25 hybrid vs dense-only
    use_fewshot: bool    # Few-shot example injection
    use_repair: bool     # Error validation & repair loop
    max_attempts: int    # Max repair attempts (default 3)
    catalog_mode: bool   # Multi-database catalog routing mode
    k_tables: int        # Retrieval table limit
    k_columns: int       # Retrieval column limit
    k_fewshots: int      # Number of few-shot examples


class AgentMetrics(TypedDict, total=False):
    """Instrumentation metrics tracked across node executions."""
    tokens_in: int
    tokens_out: int
    llm_calls: int
    per_node_latency: Dict[str, float]
    total_latency_s: float


class AgentState(TypedDict, total=False):
    """State passing through the LangGraph state machine."""
    # Inputs
    question: str
    db_id: Optional[str]
    evidence: Optional[str]
    config: AgentConfig

    # Schema & Retrieval context
    schema_text: str
    tables: List[str]
    columns: List[str]
    schema_truncated: bool
    fewshots: List[Dict[str, str]]

    # SQL Generation & Repair
    sql: str
    validation_error: Optional[str]
    exec_error: Optional[str]
    attempts: int
    history: List[Dict[str, Any]]

    # Execution outcomes
    rows: Optional[List[Any]]
    col_names: Optional[List[str]]
    row_count: int
    truncated: bool
    answer: str
    status: str  # "ok" or "failed"

    # Diagnostics & Profiling
    metrics: AgentMetrics
    clarify_needed: bool
    clarify_question: Optional[str]

