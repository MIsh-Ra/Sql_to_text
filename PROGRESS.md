# Project Progress

## Current Phase: Phase 3 (Indexes)

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
  - Implemented `eval/build_subsets.py`: created `spider_dev_200`, `bird_mini_200`, `spider_tune_50`, `bird_tune_50`, combined `tune_50`, and `catalog_dbs.json` with exact stratified difficulty distributions.
  - Implemented `app/db.py`: SQLite read-only sandbox with authorizer, timeout progress handler, and row limits.
  - Implemented `app/validate.py`: AST SQL validator with sqlglot SQLite dialect, blocking DDL/DML, stacked statements, and dangerous functions.
  - Implemented `app/prompts.py`: system and user prompts for generation, repair, and summarization, plus prompt hashing.
  - Implemented `app/state.py` and `app/graph.py`: LangGraph state machine with checkpointer and repair loop.
  - Implemented `eval/metrics.py`: execution matching, bootstrap 95% CIs, exact McNemar test, cost calculation.
  - Implemented `eval/run_eval.py`: evaluation runner with resume support, header metadata, and JSONL streaming.
  - Passed acceptance check: C0 ran on 20 tuning questions and outputted valid JSONL with all required metrics.
  - Verified 21/21 pytest tests pass and ruff is clean.

### In Progress
- [ ] **Phase 3 (Indexes)**:
  - Dense + BM25 hybrid schema index in `app/schema_index.py` (`BAAI/bge-small-en-v1.5` on CPU, Chroma persistent client, rank_bm25 with RRF k=60).
  - Compact CREATE TABLE renderer and foreign-key expansion.
  - Catalog-mode database router.
  - Few-shot index in `app/fewshot_index.py` (training questions only, strictly excluding questions from the same `db_id`).
  - Unit tests for retrieval and exclusion constraints.
  - Phase 3 acceptance check: table recall@k printed for full eval subsets, plus token reduction against full schema.

### Next
- **Phase 4 (Agent)**: Full graph verification, C4 runs on tune50, scripted tests covering all state routes.

### Known Issues
- None.
