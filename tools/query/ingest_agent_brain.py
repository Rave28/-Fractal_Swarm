import os
import sys
import uuid
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
# Targeting your specific Antigravity brain directory on Windows
BRAIN_DIR = Path(r"C:\Users\Admin\.gemini\antigravity\brain")
DB_PATH = os.path.expanduser("~/mybrain_data")
# Creating a dedicated collection so agent thoughts don't pollute live web research
COLLECTION_NAME = "antigravity_agent_history"


def chunk_text(raw_text: str, max_chunk_size: int = 1000) -> list[str]:
    """Slices the agent's stream-of-consciousness into semantic blocks."""
    paragraphs = raw_text.split("\n\n")
    chunks: list[str] = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) < max_chunk_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def main() -> None:
    if not BRAIN_DIR.exists():
        print(f"(-) Target directory not found: {BRAIN_DIR}")
        sys.exit(1)

    print(f"Connecting to MyBrain local node at {DB_PATH}...")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        print(f"(-) Failed to boot vector node: {e}")
        sys.exit(1)

    total_chunks = 0
    print(f"Scanning {BRAIN_DIR} for agent memories...")

    # Recursively grab all markdown, text, or json files the agent generates
    for filepath in BRAIN_DIR.rglob("*.*"):
        if filepath.suffix.lower() not in [".md", ".txt", ".json"]:
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
            if not content.strip():
                continue

            # CRITICAL: Prepend the filename so the vector DB knows exactly which memory this is
            enriched_content = f"Source Agent Memory File: {filepath.name}\n\n{content}"
            chunks = chunk_text(enriched_content)

            if chunks:
                chunk_ids = [f"memory_{uuid.uuid4().hex[:8]}" for _ in chunks]
                metadatas = [
                    {"source_file": filepath.name, "chunk_index": i}
                    for i in range(len(chunks))
                ]

                collection.upsert(ids=chunk_ids, documents=chunks, metadatas=metadatas)
                total_chunks += len(chunks)
                print(f"(+) Ingested {filepath.name} ({len(chunks)} chunks)")

        except Exception as e:
            print(f"(!) Skipped {filepath.name} (Unreadable formatting): {e}")

    print(
        f"\nOMNISCIENCE ACHIEVED: {total_chunks} agent thought chunks permanently wired into '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    main()
