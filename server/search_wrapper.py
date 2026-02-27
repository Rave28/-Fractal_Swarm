import sys
import json
import argparse

# --- Portable Path Discovery ---
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Swarm Root is one level up from /server/
SWARM_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BRAIN_PATH = os.path.join(SWARM_ROOT, "core", "brain", "myBrAIn")

sys.path.append(BRAIN_PATH)
from core.db import BrainDB  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    # Use the unified DB path
    db_path = os.path.join(BRAIN_PATH, "chroma_db")
    if not os.path.exists(db_path):
        db_path = os.path.expanduser("~/mybrain_data")

    db = BrainDB(db_path=db_path)
    # Search within the workbase it was ingested
    results = db.search(args.query, 工作区="vibe_backend_demo", limit=args.limit)

    out = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        ids = results["ids"][0]
        docs = results["documents"][0]
        distances = results["distances"][0]
        for i in range(len(ids)):
            # Convert euclidean distance to a mock relevance score
            relevance = max(0.0, 1.0 - distances[i])
            out.append(
                {
                    "id": ids[i],
                    "content": docs[i],
                    "relevance_score": round(relevance, 4),
                }
            )

    print(json.dumps(out))


if __name__ == "__main__":
    main()
