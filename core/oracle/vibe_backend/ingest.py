import sys
import hashlib
from pathlib import Path

# Connect to the restored mybrain vector database
sys.path.append(r"d:\Temp\myBrAIn")
from core.db import BrainDB

# ── Namespace resolver ─────────────────────────────────────────────────────────
# Import the workspace_id resolver. We walk up from this file's location to find
# a .swarm_workspace file, falling back to the directory name slug.
_NAMESPACE_DIR = Path(__file__).parent.parent.parent.parent / "core" / "namespace"
sys.path.insert(0, str(_NAMESPACE_DIR))
from workspace_id import get_workspace_id


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Split text into semantic chunks with word-boundary overflow protection."""
    chunks: list[str] = []
    current_chunk = ""
    for line in text.split("\n"):
        sub_lines: list[str] = []
        if max_chars > 0 and len(line) > max_chars:
            words = line.split(" ")
            sub = ""
            for word in words:
                if len(sub) + len(word) + 1 > max_chars and sub:
                    sub_lines.append(sub.strip())
                    sub = word + " "
                else:
                    sub += word + " "
            if sub.strip():
                sub_lines.append(sub.strip())
        else:
            sub_lines = [line]

        for sub_line in sub_lines:
            if max_chars > 0 and len(current_chunk) + len(sub_line) > max_chars:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sub_line + "\n"
            else:
                current_chunk += sub_line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def run_ingestion(workspace_override: str | None = None):
    """
    Ingest documents into MyBrAIn, tagging every chunk with the active workspace_id.

    Args:
        workspace_override: If set, bypasses the .swarm_workspace file resolver.
                            Useful for CI/CD pipelines and batch ingestion scripts.
    """
    # ── RESOLVE WORKSPACE NAMESPACE ──────────────────────────────────────────
    workspace_id = workspace_override or get_workspace_id()
    print(f"[*] Workspace namespace: '{workspace_id}'")
    # ─────────────────────────────────────────────────────────────────────────

    print("Initializing MyBrAIn Vector Engine...")
    db = BrainDB()

    # Local notes to ingest
    files_to_ingest = [
        r"C:\Users\Admin\.gemini\antigravity\brain\64bf93a9-bce7-402c-9912-d6d83b5a0fa5\vibe_coding_audit.md",
        r"C:\Users\Admin\.gemini\antigravity\brain\64bf93a9-bce7-402c-9912-d6d83b5a0fa5\walkthrough.md",
    ]

    count = 0
    print(
        f"Starting ingestion of {len(files_to_ingest)} files into namespace '{workspace_id}'..."
    )

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
            mem_id = f"context_{workspace_id}_{path.name}_{i}_{chunk_hash}"

            # ── NAMESPACE TAG injected into every metadata record ──────────
            metadata = {
                "workspace": workspace_id,  # <- The primary namespace filter key
                "workbase_id": workspace_id,  # <- Backward compat with BrainDB.search()
                "project_name": workspace_id.replace("_", " ").title(),
                "type": "context",
                "category": "documentation",
                "source": path.name,
                "source_file": path.name,
                "chunk_index": i,
            }
            # ──────────────────────────────────────────────────────────────

            db.add_memory(mem_id, chunk, metadata)
            count += 1
            print(f"  Ingested chunk {i + 1}/{len(chunks)} of {path.name}")

    print(
        f"Ingestion complete. {count} semantic chunks tagged with namespace='{workspace_id}'."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fractal Swarm — MyBrAIn Ingestion Agent"
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override workspace_id (default: auto-detect from .swarm_workspace)",
    )
    args = parser.parse_args()
    run_ingestion(workspace_override=args.workspace)
