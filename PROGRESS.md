# Project Progress

## Current Phase: Phase 2 (Data and baseline)

### Done
- [x] **Phase 1 (Setup)**:
  - Initialized Git repository and configured `.gitignore` (safeguarding API keys, datasets, caches).
  - Created virtual environment and installed all dependencies (`langchain`, `langgraph`, `chromadb`, `sentence-transformers`, `rank-bm25`, `sqlglot`, `fastapi`, `streamlit`, `pytest`, `ruff`, etc.).
  - Generated pinned `requirements.txt`.
  - Configured `.env` and `.env.example` supporting Groq, Gemini, and OpenAI.
  - Implemented `app/config.py` with Pydantic settings.
  - Implemented `app/cache.py` with deterministic SHA256 disk caching.
  - Implemented `app/llm.py` unified factory with exponential backoff on rate limits and disk caching.
  - Configured GitHub Actions CI workflow in `.github/workflows/ci.yml`.
  - Created `LICENSE` (MIT) and initialized `docs/DECISIONS.md`.
  - Passed acceptance check: unit tests pass offline, live LLM call tested and served from disk cache on second invocation.

### In Progress
- [ ] **Phase 2 (Data and baseline)**:
  - Implement `scripts/download_data.py` (Spider dev/train, BIRD mini-dev).
  - Implement `eval/build_subsets.py` (`spider_dev_200`, `bird_mini_200`, `tune_50`, and catalog DBs list).
  - Implement SQLite sandbox `app/db.py` (read-only URI, authorizer, progress handler timeout, row cap).
  - Implement AST SQL validator `app/validate.py` (sqlglot SQLite dialect, blocked statements, table and column validation).
  - Implement baseline C0 graph (`app/state.py`, `app/prompts.py`, `app/graph.py`).
  - Implement evaluation harness (`eval/metrics.py`, `eval/run_eval.py`).
  - Phase 2 acceptance check: C0 runs on 20 `tune50` questions and writes a results JSONL with every required field.

### Next
- **Phase 3 (Indexes)**: Dense + BM25 hybrid schema index (`app/schema_index.py`), few-shot index (`app/fewshot_index.py`), retrieval metrics.

### Known Issues
- None.
