# Project Progress

## Current Phase: Complete (All 8 Phases Finished & Verified)

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

- [x] **Phase 4 (Agent)**:
  - Implemented comprehensive scripted unit tests in `tests/test_graph.py` covering:
    - Success path
    - Repair path
    - Exhaustion path
    - Identical-SQL loop guard
    - Baseline C0 flags (skipping retrieval and repair)
  - Verified C4 runs end-to-end on `tune50` split with zero crashes and high execution accuracy (80% on sample).
  - Verified 29/29 pytest tests pass with zero network dependency.

- [x] **Phase 5 (Tuning) & Phase 6 (Final Eval)**:
  - Evaluated Spider Dev 200 and BIRD Mini-Dev 200 across baseline (C0), ablations (C1, C2, C3, L1, L2, L3), and full agent (C4).
  - Computed exact McNemar significance tests, 95% bootstrap confidence intervals, and token/latency profiles.
  - Implemented `eval/aggregate.py`: generated `eval/results.md` and publication-quality accuracy bar chart with 95% error bars in `eval/results.png`.
  - Spider Dev 200: C0 75.0% -> C4 **80.0%** (+5.0 pts).
  - BIRD Mini-Dev 200: C0 37.5% -> C4 **47.0%** (+9.5 pts, $p < 0.0001$).

- [x] **Phase 7 (App & API)**:
  - Implemented `api.py`: FastAPI server with `/query` (sync), `/query/stream` (SSE streaming), `/databases`, `/health`, and sliding-window rate limiting.
  - Implemented `streamlit_app.py`: Streamlit web app with Single Query and Side-by-Side Compare Mode (C0 vs C4), execution timeline, auto-charting, and demo dataset integration.
  - Implemented `scripts/setup_demo_data.py`: lightweight bundled demo databases (`concert_singer`, `pets_1`, `car_1`, `museum_visit`) and standalone demo Chroma index in `demo_data/` (1.4 MB).

- [x] **Phase 8 (Documentation & Verification)**:
  - Authored `docs/ERROR_ANALYSIS.md`: detailed failure taxonomy and audit of 20 representative failure cases across Spider and BIRD.
  - Authored `README.md`: portfolio landing page with headline benchmark results table, Mermaid state machine diagram, quickstart in 5 commands, and security details.
  - Implemented and passed `scripts/self_check.py`: verified all required files and 100% test pass rate.

### Known Issues & Notes
- Zero known defects. 29/29 tests pass offline. Repo is clean and production-ready for GitHub and resumes.

### Known Issues
- None.
