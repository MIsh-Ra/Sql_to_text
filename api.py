"""FastAPI backend service for Text-to-SQL agent with LangGraph orchestration."""

import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.graph import run as run_graph
from app.graph import stream as stream_graph
from app.state import AgentConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Portfolio Text-to-SQL Agent API",
    description="High-precision Text-to-SQL API with LangGraph self-correction, hybrid RAG, and execution sandboxing.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple sliding window in-memory rate limiter (per client IP)
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_S = 60.0
request_history: Dict[str, List[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "127.0.0.1"
    now = time.time()

    # Prune outdated timestamps
    history = request_history[client_ip]
    request_history[client_ip] = [t for t in history if now - t < RATE_LIMIT_WINDOW_S]

    if len(request_history[client_ip]) >= RATE_LIMIT_REQUESTS:
        return Response(
            content=json.dumps({"detail": "Rate limit exceeded. Max 60 requests per minute."}),
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="application/json",
        )

    request_history[client_ip].append(now)
    response = await call_next(request)
    return response


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="Natural language question to query.")
    db_id: Optional[str] = Field(None, description="Target database ID (leave null for catalog mode routing).")
    evidence: Optional[str] = Field(None, description="Domain hints or external evidence (e.g. for BIRD).")
    use_retrieval: bool = Field(True, description="Enable hybrid schema retrieval.")
    hybrid: bool = Field(True, description="Enable dense + BM25 fusion.")
    use_fewshot: bool = Field(True, description="Enable few-shot demonstration retrieval.")
    use_repair: bool = Field(True, description="Enable self-correction repair loop.")
    max_attempts: int = Field(3, ge=1, le=5, description="Maximum execution repair attempts.")
    catalog_mode: bool = Field(False, description="Enable cross-database routing.")


class QueryResponse(BaseModel):
    question: str
    db_id: Optional[str]
    sql: Optional[str]
    answer: Optional[str]
    status: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    truncated: bool
    validation_error: Optional[str]
    exec_error: Optional[str]
    attempts: int
    latency_s: float
    tokens_in: int
    tokens_out: int
    llm_calls: int


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Service health and model metadata."""
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "chroma_dir": str(settings.chroma_dir),
    }


@app.get("/databases")
def list_databases() -> Dict[str, Any]:
    """List available indexed SQLite databases."""
    databases = []

    # Check demo directory
    demo_dir = Path("./demo_data")
    if demo_dir.exists():
        for f in demo_dir.glob("*.sqlite"):
            databases.append({"db_id": f.stem, "source": "demo", "size_kb": round(f.stat().st_size / 1024, 1)})

    # Check spider databases
    spider_dir = Path("./data/spider/database")
    if spider_dir.exists():
        for d in spider_dir.iterdir():
            if d.is_dir() and (d / f"{d.name}.sqlite").exists():
                databases.append({"db_id": d.name, "source": "spider", "size_kb": round((d / f"{d.name}.sqlite").stat().st_size / 1024, 1)})

    return {"count": len(databases), "databases": databases}


@app.post("/query", response_model=QueryResponse)
def execute_query_endpoint(req: QueryRequest) -> QueryResponse:
    """Execute Text-to-SQL query synchronously with full state machine."""
    config: AgentConfig = {
        "use_retrieval": req.use_retrieval,
        "hybrid": req.hybrid,
        "use_fewshot": req.use_fewshot,
        "use_repair": req.use_repair,
        "max_attempts": req.max_attempts,
        "catalog_mode": req.catalog_mode if req.db_id is None else False,
    }

    start = time.perf_counter()
    try:
        final_state = run_graph(
            question=req.question,
            db_id=req.db_id,
            evidence=req.evidence,
            config=config,
        )
    except Exception as exc:
        logger.exception("Error executing agent graph: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    latency = time.perf_counter() - start
    metrics = final_state.get("metrics", {})

    return QueryResponse(
        question=req.question,
        db_id=final_state.get("db_id"),
        sql=final_state.get("sql"),
        answer=final_state.get("answer"),
        status=final_state.get("status", "failed"),
        columns=final_state.get("col_names", []),
        rows=[list(r) for r in final_state.get("rows", [])],
        row_count=final_state.get("row_count", 0),
        truncated=final_state.get("truncated", False),
        validation_error=final_state.get("validation_error"),
        exec_error=final_state.get("exec_error"),
        attempts=final_state.get("attempts", 0),
        latency_s=round(latency, 3),
        tokens_in=metrics.get("tokens_in", 0),
        tokens_out=metrics.get("tokens_out", 0),
        llm_calls=metrics.get("llm_calls", 0),
    )


@app.post("/query/stream")
async def execute_query_stream(req: QueryRequest):
    """Execute Text-to-SQL query streaming state updates via Server-Sent Events (SSE)."""
    config: AgentConfig = {
        "use_retrieval": req.use_retrieval,
        "hybrid": req.hybrid,
        "use_fewshot": req.use_fewshot,
        "use_repair": req.use_repair,
        "max_attempts": req.max_attempts,
        "catalog_mode": req.catalog_mode if req.db_id is None else False,
    }

    async def event_generator() -> AsyncGenerator[str, None]:
        for node_name, node_update in stream_graph(
            question=req.question,
            db_id=req.db_id,
            evidence=req.evidence,
            config=config,
        ):
            event_data = {
                "node": node_name,
                "status": node_update.get("status"),
                "sql": node_update.get("sql"),
                "attempts": node_update.get("attempts"),
                "row_count": node_update.get("row_count"),
                "answer": node_update.get("answer"),
                "error": node_update.get("validation_error") or node_update.get("exec_error"),
            }
            yield f"data: {json.dumps(event_data)}\n\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
