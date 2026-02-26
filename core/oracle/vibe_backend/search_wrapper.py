import sys
import json
import argparse

# Connect to the restored mybrain vector database
sys.path.append(r"d:\Temp\myBrAIn")
from core.db import BrainDB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    db = BrainDB()
    # Search within the workbase it was ingested
    results = db.search(args.query, workbase_id="vibe_backend_demo", limit=args.limit)

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
