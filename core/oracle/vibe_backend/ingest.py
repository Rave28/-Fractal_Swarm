import sys
import hashlib
from pathlib import Path

# Connect to the restored mybrain vector database
sys.path.append(r"d:\Temp\myBrAIn")
from core.db import BrainDB


def chunk_text(text, max_chars=800):
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) > max_chars:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def run_ingestion():
    print("Initializing MyBrAIn Vector Engine...")
    db = BrainDB()
    wb_id = "vibe_backend_demo"

    # Local notes to ingest
    files_to_ingest = [
        r"C:\Users\Admin\.gemini\antigravity\brain\64bf93a9-bce7-402c-9912-d6d83b5a0fa5\vibe_coding_audit.md",
        r"C:\Users\Admin\.gemini\antigravity\brain\64bf93a9-bce7-402c-9912-d6d83b5a0fa5\walkthrough.md",
    ]

    count = 0
    print(f"Starting ingestion of {len(files_to_ingest)} files...")

    for file_path in files_to_ingest:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found at {file_path}")
            continue

        content = path.read_text(encoding="utf-8")
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            mem_id = f"context_{wb_id}_{path.name}_{i}_{chunk_hash}"
            metadata = {
                "workbase_id": wb_id,
                "project_name": "Vibe Backend",
                "type": "context",
                "category": "documentation",
                "source": path.name,
            }

            # Upsert memory
            db.add_memory(mem_id, chunk, metadata)
            count += 1
            print(f" Ingested chunk {i + 1}/{len(chunks)} of {path.name}")

    print(f"Ingestion complete. {count} semantic chunks added to mybrain!")


if __name__ == "__main__":
    run_ingestion()
