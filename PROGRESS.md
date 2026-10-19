# Project Progress

## Current Phase: Phase 4 (Agent)

### Done
- [x] **Phase 1 (Setup)**:
  - Initialized Git repository and configured `.gitignore` (safeguarding API keys, datasets, caches).
  - Created virtual environment and installed dependencies.
  - Generated pinned `requirements.txt`.
  - Configured `.env` and `.env.example`.
  - Implemented `app/config.py` with Pydantic settings.
  - Implemented `app/cache.py` with deterministic SHA256 disk caching.
  - Implemented `app/llm.py` unified factory with exponential backoff on rate limits and disk caching.
  - Configured GitHub Actions CI workflow in `.github/workflows/ci.yml`.
  - Created `LICENSE` (MIT) and initialized `docs/DECISIONS.md`.
  - Passed acceptance check: unit tests pass offline, live LLM call tested and served from disk cache.

- [x] **Phase 2 (Data and baseline)**:
  - Implemented `scripts/download_data.py`: downloaded and verified Spider dev/train and BIRD mini-dev datasets and databases.
  - Implemented `eval/build_subsets.py`: created `spider_dev_200`, `bird_mini_200`, `spider_tune_50`, `bird_tune_50`, combined `tune_50`, and `catalog_dbs.json`.
  - Implemented `app/db.py`: SQLite read-only sandbox with authorizer, timeout progress handler, and row limits.
  - Implemented `app/validate.py`: AST SQL validator with sqlglot SQLite dialect, blocking DDL/DML, stacked statements, and dangerous functions.
  - Implemented `app/prompts.py`: system and user prompts for generation, repair, and summarization, plus prompt hashing.
  - Implemented `app/state.py` and `app/graph.py`: LangGraph state machine with checkpointer and repair loop.
  - Implemented `eval/metrics.py`: execution matching, bootstrap 95% CIs, exact McNemar test, cost calculation.
  - Implemented `eval/run_eval.py`: evaluation runner with resume support, header metadata, and JSONL streaming.
  - Passed acceptance check: C0 ran on 20 tuning questions and outputted valid JSONL with all required metrics.

- [x] **Phase 3 (Indexes)**:
  - Implemented `app/schema_index.py`: dense (`BAAI/bge-small-en-v1.5` on CPU) + BM25 hybrid schema index with Reciprocal Rank Fusion ($k=60$), foreign-key expansion, and compact CREATE TABLE renderer.
  - Implemented `app/fewshot_index.py`: 1,000 embedded training questions with strict target database exclusion.
  - Implemented `scripts/build_indexes.py`: indexed all 20 Spider dev databases and 11 BIRD dev databases into persistent Chroma client.
  - Passed acceptance check:
    - Spider Dev 200: **100.00% Table Recall@k**.
    - BIRD Mini 200: **99.37% Table Recall@k**, with **26.03% Token Reduction** against full schema.
  - Verified 24/24 pytest tests pass and ruff is clean.

### In Progress
- [ ] **Phase 4 (Agent)**:
  - Scripted unit tests in `tests/test_graph.py` with fake LLM covering:
    - Success path (direct generate -> validate -> execute -> summarize)
    - Repair path (failing SQL repaired and re-validated)
    - Exhaustion path (repair attempts reach max_attempts -> fail)
    - Identical-SQL loop guard (repair returns identical SQL -> immediate fail)
    - Baseline flags (skipping retrieval and repair in C0)
  - Run C4 on `tune50` questions with zero crashes.
  - Acceptance check: C4 runs smoothly on tuning questions, and scripted tests cover every graph route.

### Next
- **Phase 5 (Tuning)**: Prompt refinement on `tune50` split only; freeze prompts and record SHA256 prompt hash.
- **Phase 6 (Final eval)**: Run full benchmarks C0-C4 and L1-L3 on both `eval200` subsets, catalog mode comparison, and aggregate results.

### Known Issues
- None.
