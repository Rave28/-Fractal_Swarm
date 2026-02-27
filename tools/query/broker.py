"""
Knowledge Broker API.
Autonomous backend endpoint that pulls dynamic search logic from
Perplexity Ask and explicitly inserts it into the local mybrain vector store.
"""

import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

# --- Configuration ---
DB_PATH = os.path.expanduser("~/mybrain_data")
COLLECTION_NAME = "knowledge_broker_collection"

# Global Variables
chroma_client: chromadb.PersistentClient | None = None
collection: chromadb.Collection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages the startup and shutdown execution of the ChromaDB Persistent Client."""
    global chroma_client, collection

    print(f"Connecting to MyBrain local ChromaDB node at: {DB_PATH}", file=sys.stderr)
    try:
        from chromadb.utils import embedding_functions

        chroma_client = chromadb.PersistentClient(path=DB_PATH)

        # Using the same `all-MiniLM-L6-v2` embedding standard model for alignment
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
        print("Knowledge Broker Database Online.", file=sys.stderr)
    except Exception as e:
        print(f"Failed to bootstrap ChromaDB PersistentClient: {e}", file=sys.stderr)
        # Allows FastAPI to continue even if db errors so we can debug, though not ideal.

    yield

    print("Shutting down Knowledge Broker orchestration...", file=sys.stderr)


# --- Core FastAPI App ---
app = FastAPI(
    title="Autonomous Knowledge Broker",
    description="Microservice built to funnel autonomous research into local vector RAM.",
    version="1.0.0",
    lifespan=lifespan,
)


class ResearchRequest(BaseModel):
    query: str
    model_config = ConfigDict(strict=True)


class ResearchResponse(BaseModel):
    status: str
    query: str
    embedded_chunks: int
    execution_time_ms: float
    model_config = ConfigDict(strict=True)


def simulate_perplexity_research(query: str) -> str:
    """
    Simulate fetching massive intelligent payloads via Perplexity MCP.
    """
    time.sleep(1.5)  # Simulate network hop and reasoning wait stream
    return (
        f"Autonomous Action Plan for '{query}':\n\n"
        "State-of-the-art backend systems employ localized vector retrieval loops. "
        "Running this orchestration natively on Python 3.12 avoids Pydantic v1 core dumps "
        "inside of external packages. Always explicitly structure ingestion parameters before sending."
    )


def chunk_intelligence(raw_text: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Segment the massive payload into semantically dense chunks.
    Avoids shattering sentences by splitting via paragraph heuristics first.
    """
    paragraphs = raw_text.split("\n\n")
    chunks: list[str] = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


@app.post("/research", response_model=ResearchResponse)
async def trigger_research(request: ResearchRequest) -> ResearchResponse:
    """
    The orchestrator.
    Takes a query string -> Triggers external LLM research -> Chunks it -> Streams to disk.
    """
    start_time = time.time()

    if not collection:
        raise HTTPException(
            status_code=503,
            detail="Local ChromaDB collection instance not properly initialized.",
        )

    # 1. Fetch Real-time Context (simulated via MCP)
    try:
        raw_intelligence = simulate_perplexity_research(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Perplexity Search MCP Error: {e}") from e

    # 2. Slice and Dice
    chunks = chunk_intelligence(raw_intelligence)
    if not chunks:
        raise HTTPException(
            status_code=400, detail="Retrieved intelligence was empty or uncategorizable."
        )

    # 3. Synchronize
    try:
        chunk_ids = [f"research_{uuid.uuid4().hex[:8]}" for _ in chunks]
        metadatas = [{"query": request.query, "chunk_index": i} for i in range(len(chunks))]

        collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            metadatas=metadatas,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to upsert logic blocks to disk: {e}"
        ) from e

    execution_time = (time.time() - start_time) * 1000

    return ResearchResponse(
        status="success",
        query=request.query,
        embedded_chunks=len(chunks),
        execution_time_ms=round(execution_time, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("broker:app", host="0.0.0.0", port=8000, reload=True)
