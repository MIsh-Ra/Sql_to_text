# Architectural & Engineering Decisions

This document records key design choices, library adaptations, and trade-offs made during the development of the Text-to-SQL agent.

---

### DEC-001: Multi-Provider LLM Factory with Universal Chat Interface
- **Context**: The system must evaluate models deterministically and support Groq, Google Gemini, and OpenAI while strictly prohibiting provider SDK imports outside `app/llm.py`.
- **Decision**: Implemented `app/llm.py` using LangChain's `init_chat_model` / `ChatGroq` / `ChatGoogleGenerativeAI` / `ChatOpenAI`. Temperature is locked at `0.0` for all evaluation runs.
- **Consequences**: Switching models only requires setting `LLM_PROVIDER` and `LLM_MODEL` in `.env` or runtime arguments, with zero code changes across the agent graph or eval harness.

### DEC-002: Deterministic Disk Caching Keyed by Prompt Hash
- **Context**: Rerunning evaluations across 200 questions with ablations would be slow and costly if identical prompts hit the live API repeatedly.
- **Decision**: Built a custom file-based disk cache in `app/cache.py` keyed by `(provider, model, temperature, sha256(prompt))`. When enabled (`LLM_CACHE_ENABLED=true`), cache hits return the stored `AIMessage` immediately with zero network latency and token cost. An environment variable allows bypassing cache when needed.

### DEC-003: SQLite Read-Only Sandbox & AST Verification
- **Context**: Generated SQL must be treated as untrusted input.
- **Decision**: Applied defense-in-depth:
  1. SQLGlot AST validation (`app/validate.py`) enforcing single SELECT/CTE statements, table existence, and blocking all DDL, DML, ATTACH, PRAGMA, and comment stacking.
  2. SQLite connection sandbox (`app/db.py`) opening databases with `file:...uri=true&mode=ro`, installing an authorizer denying anything other than SELECT/READ/FUNCTION/RECURSIVE, setting progress handlers for execution timeout, and capping row outputs.

### DEC-004: Sentence-Transformers Local Embedding Model
- **Context**: Schema and few-shot vector indexes require CPU-friendly local embeddings without cloud API rate limits or costs.
- **Decision**: Used `BAAI/bge-small-en-v1.5` on CPU via `sentence-transformers` and Chroma persistent local client.


### DEC-005: Model Selection for Groq (`openai/gpt-oss-120b`)
- **Context**: On Groq, `llama-3.3-70b-versatile` returned model not found under the active API key permissions. Querying the Groq `/models` endpoint revealed `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, and `qwen/qwen3.8-27b`.
- **Decision**: Selected `openai/gpt-oss-120b` on Groq as the primary model. It delivers high-capacity reasoning (120B parameters) and sub-second generation speeds.

### DEC-006: Official Spider Component Hardness Calculation
- **Context**: Spider `dev.json` does not include explicit difficulty labels in its raw JSON files, unlike BIRD.
- **Decision**: Implemented the exact official Spider hardness calculation (`count_component1`, `count_component2`, `count_others`) based on the official AST component evaluator from Yale LILY. This ensures stratified sampling for `spider_dev_200` exactly mirrors the difficulty distribution of the full dev benchmark.

### DEC-007: Defense-in-Depth Execution Sandbox
- **Context**: Executing LLM-generated SQL against SQLite must be safe from file modification, malicious extensions, infinite recursive loops, or PRAGMA attacks.
- **Decision**: Opened databases via `file:///path?mode=ro`, installed `set_authorizer` granting solely `SQLITE_SELECT`, `SQLITE_READ`, `SQLITE_FUNCTION`, and `SQLITE_RECURSIVE`, and installed a progress handler checking timeouts every 500 virtual machine instructions. Fetched rows are capped without modifying SQL AST.
