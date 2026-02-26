from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import httpx
import os
import sys
import uuid
import chromadb
from chromadb.utils import embedding_functions

# Initialize Vector DB connection for permanent memory hoarding
CONFIDENCE_THRESHOLD = 0.45
db_path = os.path.expanduser("~/mybrain_data")
chroma_client = chromadb.PersistentClient(path=db_path)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_or_create_collection(
    name="knowledge_broker_collection", embedding_function=emb_fn
)

# --- Mock Knowledge Retrieval Module (Synthesized from mybrain/DeepCode) ---
# Concept: Vector Semantic Search & Embeddings
# Best Practice: FastAPI Dependency Injection for "Search Engine" lifetime management

app = FastAPI(
    title="Vibe Backend: Pro-Level Knowledge Engine",
    description="FastAPI + UV + Ruff + Autonomous Knowledge Synthesis",
    version="1.0.0",
)


# Models
class SearchResult(BaseModel):
    id: int
    content: str
    relevance_score: float


class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = Field(default=5, ge=1, le=20)


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    status: str
    query: str
    embedded_chunks: int
    execution_time_ms: float


# Real Vector Engine Integration via Subprocess
class VectorSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.is_loaded = True

    async def semantic_search(self, query: str, limit: int) -> List[SearchResult]:
        import subprocess
        import json

        cmd = [
            r"d:\Temp\myBrAIn\venv\Scripts\python.exe",
            r"d:\Temp\vibe_backend\search_wrapper.py",
            query,
            "--limit",
            str(limit),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)

        try:
            data = json.loads(result.stdout)
            return [
                SearchResult(
                    id=i,
                    content=item["content"],
                    relevance_score=item["relevance_score"],
                )
                for i, item in enumerate(data)
            ]
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse search results: {e}. Output was {result.stdout[:100]}",
            )


# Dependency Injection Pattern (Retrieved from deepcode-brain)
async def get_search_engine() -> VectorSearchEngine:
    """Dependency that ensures engine availability for search endpoints."""
    engine = VectorSearchEngine()
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Vector engine not initialized")
    return engine


# Endpoints
@app.get("/health", tags=["System"])
async def health_check():
    """Vibe Flow Health Check: High Autonomy Status"""
    return {
        "status": "online",
        "engine": "uv + fast-ready",
        "timestamp": time.time(),
        "vibe_optimized": True,
    }


@app.get("/search", tags=["Knowledge"], response_model=List[SearchResult])
async def search(
    query: str = Query(..., min_length=2),
    limit: int = 5,
    engine: VectorSearchEngine = Depends(get_search_engine),
):
    """
    Knowledge retrieval endpoint.
    Utilizes localized embedding search logic retrieved via mybrain.
    """
    return await engine.semantic_search(query, limit)


async def execute_live_perplexity_research(query: str) -> str:
    """
    Query the live Perplexity 'sonar-pro' model for real-time web research.
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="PERPLEXITY_API_KEY environment variable is missing. Check your terminal session.",
        )

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are a Lead Data Engineer. Provide concise, highly technical, and up-to-date architectural best practices based on live web data. Output pure, dense information without conversational filler.",
            },
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,  # Low temperature for highly deterministic, factual output
    }

    # Asynchronous context manager to prevent blocking the FastAPI event loop
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Perplexity API Error: {e.response.text}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Network Error: {str(e)}")


def chunk_intelligence(text: str, max_chars: int = 800) -> List[str]:
    """Slice and dice raw web intelligence into semantic blocks."""
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) > max_chars:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


@app.post("/research", response_model=ResearchResponse, tags=["Research"])
async def trigger_research(request: ResearchRequest) -> ResearchResponse:
    """
    The Tiered Orchestrator.
    1. Checks local memory for high-confidence match.
    2. If cache miss, strikes Perplexity API, chunks, and memorizes.
    """
    start_time = time.time()

    if not collection:
        raise HTTPException(
            status_code=503,
            detail="Local ChromaDB collection instance offline. Memory bank unavailable.",
        )

    # --- TIER 1: Semantic Cache Check (Cost & Latency Saver) ---
    cache_results = collection.query(
        query_texts=[request.query],
        n_results=1,  # We only need the absolute best match to verify confidence
    )

    if cache_results["documents"] and cache_results["documents"][0]:
        best_distance = cache_results["distances"][0][0]
        if best_distance <= CONFIDENCE_THRESHOLD:
            # We already know this! Serve from local memory.
            execution_time = (time.time() - start_time) * 1000
            print(
                f"⚡ CACHE HIT: Served '{request.query}' from local memory. (Distance: {best_distance:.4f})",
                file=sys.stderr,
            )

            return ResearchResponse(
                status="local_cache_hit",
                query=request.query,
                embedded_chunks=0,  # Nothing new to embed
                execution_time_ms=round(execution_time, 2),
            )

    print(
        f"⚠️ CACHE MISS: Novel concept detected. Routing to Perplexity Sonar-Pro...",
        file=sys.stderr,
    )

    # --- TIER 2: Fetch Live Context (Sonar-Pro API) ---
    try:
        raw_intelligence = await execute_live_perplexity_research(request.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Oracle API Error: {e}") from e

    # Slice and Dice into Semantic Blocks
    chunks = chunk_intelligence(raw_intelligence)
    if not chunks:
        raise HTTPException(
            status_code=400, detail="Retrieved intelligence was uncategorizable."
        )

    # Synchronize to Local Memory
    try:
        chunk_ids = [f"mem_{uuid.uuid4().hex[:8]}" for _ in chunks]
        metadatas = [
            {"query": request.query, "chunk_index": i} for i in range(len(chunks))
        ]

        collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            metadatas=metadatas,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to commit logic to local disk: {e}"
        ) from e

    execution_time = (time.time() - start_time) * 1000

    return ResearchResponse(
        status="live_research_memorized",
        query=request.query,
        embedded_chunks=len(chunks),
        execution_time_ms=round(execution_time, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
