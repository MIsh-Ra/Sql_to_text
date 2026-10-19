"""Portfolio Streamlit Demo for Hybrid Text-to-SQL Agent with LangGraph Orchestration."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from app.config import settings
from app.graph import run as run_graph
from app.state import AgentConfig

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Text-to-SQL Portfolio Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #4f46e5;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-ok { background-color: #dcfce7; color: #166534; }
    .badge-failed { background-color: #fee2e2; color: #991b1b; }
    .badge-repair { background-color: #fef3c7; color: #92400e; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Demo Questions
DEMO_QUESTIONS_PATH = Path("./demo_data/sample_questions.json")
if DEMO_QUESTIONS_PATH.exists():
    with open(DEMO_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        SAMPLE_QUESTIONS = json.load(f)
else:
    SAMPLE_QUESTIONS = {
        "concert_singer": [
            "How many singers do we have?",
            "What is the average, minimum, and maximum age of all singers?",
            "List the names of singers who have performed in more than one concert.",
            "Show the stadium names and their capacities where concerts took place in 2014.",
        ],
        "pets_1": [
            "Find the first name and age of students who have a pet.",
            "How many students do not have any pets?",
            "What is the average weight of pets by pet type?",
        ],
        "car_1": [
            "How many cars were produced with 4 cylinders?",
            "What is the average horsepower across all car models?",
            "Show the car names and their MPG with the highest fuel efficiency.",
        ],
        "museum_visit": [
            "List all museum names ordered by their open year.",
            "How many visitors attended each museum in 2018?",
        ],
    }


def get_available_dbs() -> List[str]:
    dbs = []
    # Check demo_data first
    demo_dir = Path("./demo_data")
    if demo_dir.exists():
        for f in demo_dir.glob("*.sqlite"):
            dbs.append(f.stem)
    if not dbs:
        spider_dir = Path("./data/spider/database")
        if spider_dir.exists():
            for d in spider_dir.iterdir():
                if d.is_dir() and (d / f"{d.name}.sqlite").exists():
                    dbs.append(d.name)
    return sorted(list(set(dbs))) or ["concert_singer", "pets_1", "car_1", "museum_visit"]


# Sidebar Configurations
with st.sidebar:
    st.title("⚡ Text-to-SQL Agent")
    st.caption("LangGraph State Machine • Hybrid Retrieval • Sandboxed Execution")

    st.subheader("Database Target")
    all_dbs = get_available_dbs()
    selected_db = st.selectbox("Select Database", all_dbs, index=0)

    st.subheader("Mode & Configuration")
    app_mode = st.radio("Execution Mode", ["Single Query", "Compare Mode (C0 Baseline vs C4 Full Agent)"])

    with st.expander("Feature Flags & Hyperparameters", expanded=False):
        use_retrieval = st.checkbox("Hybrid Schema Retrieval", value=True)
        hybrid_fuse = st.checkbox("Dense + BM25 (RRF k=60)", value=True, disabled=not use_retrieval)
        use_fewshot = st.checkbox("Few-Shot Injection (strict db-exclusion)", value=True)
        use_repair = st.checkbox("Self-Correction Repair Loop", value=True)
        max_attempts = st.slider("Max Repair Attempts", 1, 5, 3, disabled=not use_repair)
        catalog_mode = st.checkbox("Cross-Database Catalog Routing", value=False)

    st.divider()
    st.caption(f"**Provider**: `{settings.llm_provider}`")
    st.caption(f"**Model**: `{settings.llm_model}`")
    st.caption("**Sandbox**: SQLite `mode=ro` + AST qualification")


# Main Interface
st.header("⚡ Enterprise Text-to-SQL Natural Language Interface")
st.markdown(
    "Ask complex analytical questions against relational databases with grounded retrieval, AST validation, and automated error recovery."
)

# Suggested queries
preset_questions = SAMPLE_QUESTIONS.get(selected_db, [])
col_q1, col_q2 = st.columns([3, 1])

with col_q2:
    if preset_questions:
        selected_preset = st.selectbox("Sample Questions", ["(Custom Input)"] + preset_questions)
    else:
        selected_preset = "(Custom Input)"

with col_q1:
    default_text = "" if selected_preset == "(Custom Input)" else selected_preset
    user_query = st.text_input("Enter natural language query:", value=default_text, placeholder="e.g. Which singers are older than 25?")

evidence_hint = st.text_input("Optional Evidence / Schema Hint (e.g. for BIRD benchmarks):", placeholder="e.g. age refers to (2024 - birth_year)")

run_btn = st.button("Generate & Execute SQL", type="primary", use_container_width=True)


def display_result_card(state: Dict[str, Any], title: str = "Query Result"):
    st.subheader(title)
    status_str = state.get("status", "failed")
    badge_cls = "badge-ok" if status_str == "ok" else "badge-failed"

    st.markdown(
        f'<span class="status-badge {badge_cls}">Status: {status_str.upper()}</span> &nbsp; '
        f'<b>Attempts</b>: {state.get("attempts", 1)} &nbsp; | &nbsp; '
        f'<b>LLM Calls</b>: {state.get("metrics", {}).get("llm_calls", 1)} &nbsp; | &nbsp; '
        f'<b>Rows</b>: {state.get("row_count", 0)}',
        unsafe_allow_html=True,
    )

    # SQL Block
    sql = state.get("sql", "")
    if sql:
        st.code(sql, language="sql")
    else:
        st.warning("No valid SQL generated.")

    # Execution error / Validation error
    if state.get("validation_error"):
        st.error(f"AST Validation Error: {state['validation_error']}")
    if state.get("exec_error"):
        st.error(f"Sandbox Execution Error: {state['exec_error']}")

    # Results Table
    rows = state.get("rows", [])
    cols = state.get("col_names", [])
    if rows and cols:
        df = pd.DataFrame(rows, columns=cols)
        st.dataframe(df, use_container_width=True)

        # Auto-chart if numeric columns exist
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) > 0 and len(df) > 1 and len(df) <= 50:
            with st.expander("📊 Auto-Generated Visualization"):
                st.bar_chart(df.set_index(df.columns[0])[numeric_cols[0]])

    # Natural Language Answer
    if state.get("answer"):
        st.info(f"💡 **Executive Summary**: {state['answer']}")


if run_btn and user_query:
    if app_mode == "Single Query":
        config: AgentConfig = {
            "use_retrieval": use_retrieval,
            "hybrid": hybrid_fuse,
            "use_fewshot": use_fewshot,
            "use_repair": use_repair,
            "max_attempts": max_attempts,
            "catalog_mode": catalog_mode if not selected_db else False,
        }

        with st.spinner("Orchestrating agent state machine..."):
            final_state = run_graph(
                question=user_query,
                db_id=None if catalog_mode else selected_db,
                evidence=evidence_hint,
                config=config,
            )

        display_result_card(final_state, title="Full Agent Result")

    else:
        # Compare Mode
        st.subheader("⚖️ Side-by-Side Comparison: Baseline (C0) vs Full Agent (C4)")
        col_c0, col_c4 = st.columns(2)

        config_c0: AgentConfig = {
            "use_retrieval": False,
            "hybrid": False,
            "use_fewshot": False,
            "use_repair": False,
            "max_attempts": 1,
            "catalog_mode": False,
        }

        config_c4: AgentConfig = {
            "use_retrieval": True,
            "hybrid": True,
            "use_fewshot": True,
            "use_repair": True,
            "max_attempts": 3,
            "catalog_mode": False,
        }

        with st.spinner("Running baseline C0 vs agent C4..."):
            state_c0 = run_graph(
                question=user_query,
                db_id=selected_db,
                evidence=evidence_hint,
                config=config_c0,
            )
            state_c4 = run_graph(
                question=user_query,
                db_id=selected_db,
                evidence=evidence_hint,
                config=config_c4,
            )

        with col_c0:
            display_result_card(state_c0, title="Naive Single-Prompt Baseline (C0)")
        with col_c4:
            display_result_card(state_c4, title="Full Agent with Repair Loop (C4)")

# Benchmark Tab
st.divider()
with st.expander("📈 Benchmark Performance & Empirical Ablations", expanded=False):
    results_md_path = Path("./eval/results.md")
    if results_md_path.exists():
        with open(results_md_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Run `python eval/aggregate.py` to view empirical benchmark results.")
