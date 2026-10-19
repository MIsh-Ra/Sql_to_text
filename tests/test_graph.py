"""Unit tests for LangGraph routing with a scripted fake LLM (no network)."""

import sqlite3
from pathlib import Path
from typing import Any, List, Optional
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph import build_graph
from app.state import AgentConfig, AgentState


class ScriptedFakeLLM(BaseChatModel):
    """Deterministic scripted LLM yielding pre-programmed responses."""

    responses: List[str]
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake-llm"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.call_count < len(self.responses):
            resp_text = self.responses[self.call_count]
        else:
            resp_text = "SELECT 1;"
        self.call_count += 1
        msg = AIMessage(
            content=resp_text,
            response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def graph_test_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Create test SQLite database and mock get_db_path."""
    db_file = tmp_path / "mock_store.sqlite"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT);")
    cur.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');")
    conn.commit()
    conn.close()

    monkeypatch.setattr("app.db.get_db_path", lambda db_id: db_file)
    monkeypatch.setattr("app.graph.get_db_path", lambda db_id: db_file)
    return "mock_store"


def test_graph_success_path(graph_test_db: str, monkeypatch: pytest.MonkeyPatch):
    """Test standard success: generate -> validate -> execute -> summarize."""
    fake_llm = ScriptedFakeLLM(responses=[
        "```sql\nSELECT name FROM users;\n```",
        "Alice and Bob are the users.",
    ])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    graph = build_graph(checkpointer=saver)

    config: AgentConfig = {
        "use_retrieval": False,
        "hybrid": False,
        "use_fewshot": False,
        "use_repair": True,
        "max_attempts": 3,
    }
    init_state: AgentState = {
        "question": "List all user names",
        "db_id": graph_test_db,
        "config": config,
        "metrics": {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0, "per_node_latency": {}},
    }

    res = graph.invoke(init_state, config={"configurable": {"thread_id": "thread-1"}})
    assert res["status"] == "ok"
    assert res["attempts"] == 1
    assert len(res["rows"]) == 2
    assert res["rows"][0] == ("Alice",)
    assert "Alice and Bob" in res["answer"]


def test_graph_repair_path(graph_test_db: str, monkeypatch: pytest.MonkeyPatch):
    """Test repair loop: attempt 1 invalid -> repair -> attempt 2 valid -> success."""
    fake_llm = ScriptedFakeLLM(responses=[
        "```sql\nSELECT non_existent_col FROM users;\n```",  # Failing SQL
        "```sql\nSELECT name FROM users;\n```",              # Repaired SQL
        "Found 2 users.",
    ])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    graph = build_graph(checkpointer=saver)

    config: AgentConfig = {
        "use_retrieval": False,
        "hybrid": False,
        "use_fewshot": False,
        "use_repair": True,
        "max_attempts": 3,
    }
    init_state: AgentState = {
        "question": "List users",
        "db_id": graph_test_db,
        "config": config,
        "metrics": {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0, "per_node_latency": {}},
    }

    res = graph.invoke(init_state, config={"configurable": {"thread_id": "thread-2"}})
    assert res["status"] == "ok"
    assert res["attempts"] == 2
    assert len(res["history"]) == 1
    assert "non_existent_col" in res["history"][0]["sql"]
    assert len(res["rows"]) == 2


def test_graph_exhaustion_path(graph_test_db: str, monkeypatch: pytest.MonkeyPatch):
    """Test repair exhaustion: repeated invalid SQL reaches max_attempts -> fail."""
    fake_llm = ScriptedFakeLLM(responses=[
        "```sql\nSELECT bad1 FROM users;\n```",
        "```sql\nSELECT bad2 FROM users;\n```",
        "```sql\nSELECT bad3 FROM users;\n```",
    ])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    graph = build_graph(checkpointer=saver)

    config: AgentConfig = {
        "use_retrieval": False,
        "hybrid": False,
        "use_fewshot": False,
        "use_repair": True,
        "max_attempts": 3,
    }
    init_state: AgentState = {
        "question": "List users",
        "db_id": graph_test_db,
        "config": config,
        "metrics": {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0, "per_node_latency": {}},
    }

    res = graph.invoke(init_state, config={"configurable": {"thread_id": "thread-3"}})
    assert res["status"] == "failed"
    assert res["attempts"] == 3
    assert len(res["history"]) == 2


def test_graph_identical_sql_loop_guard(graph_test_db: str, monkeypatch: pytest.MonkeyPatch):
    """Test loop guard: repair returns identical SQL -> immediate fail."""
    fake_llm = ScriptedFakeLLM(responses=[
        "```sql\nSELECT same_bad_query FROM users;\n```",
        "```sql\nSELECT same_bad_query FROM users;\n```",  # Identical SQL
    ])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    graph = build_graph(checkpointer=saver)

    config: AgentConfig = {
        "use_retrieval": False,
        "hybrid": False,
        "use_fewshot": False,
        "use_repair": True,
        "max_attempts": 3,
    }
    init_state: AgentState = {
        "question": "List users",
        "db_id": graph_test_db,
        "config": config,
        "metrics": {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0, "per_node_latency": {}},
    }

    res = graph.invoke(init_state, config={"configurable": {"thread_id": "thread-4"}})
    assert res["status"] == "failed"
    # Should stop on attempt 2 without trying a 3rd attempt
    assert res["attempts"] == 2


def test_graph_baseline_flags(graph_test_db: str, monkeypatch: pytest.MonkeyPatch):
    """Test C0 baseline: use_repair=False immediately fails on error without repairing."""
    fake_llm = ScriptedFakeLLM(responses=[
        "```sql\nSELECT bad_query FROM users;\n```",
    ])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    graph = build_graph(checkpointer=saver)

    # C0: use_repair is False
    config: AgentConfig = {
        "use_retrieval": False,
        "hybrid": False,
        "use_fewshot": False,
        "use_repair": False,
        "max_attempts": 1,
    }
    init_state: AgentState = {
        "question": "List users",
        "db_id": graph_test_db,
        "config": config,
        "metrics": {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0, "per_node_latency": {}},
    }

    res = graph.invoke(init_state, config={"configurable": {"thread_id": "thread-5"}})
    assert res["status"] == "failed"
    assert res["attempts"] == 1
    assert len(res["history"]) == 0
    assert fake_llm.call_count == 1
