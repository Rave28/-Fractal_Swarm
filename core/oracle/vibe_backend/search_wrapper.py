import sys
import json
import argparse
from pathlib import Path

# Connect to the restored mybrain vector database
sys.path.append(r"d:\Temp\myBrAIn")
from core.db import BrainDB

# ── Namespace resolver ─────────────────────────────────────────────────────────
_NAMESPACE_DIR = Path(__file__).parent.parent.parent.parent / "core" / "namespace"
sys.path.insert(0, str(_NAMESPACE_DIR))
from workspace_id import get_workspace_id


def main():
    parser = argparse.ArgumentParser(
        description="Fractal Swarm — Vector Search Wrapper"
    )
    parser.add_argument("query", help="Semantic search query string")
    parser.add_argument("--limit", type=int, default=5, help="Max results to return")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace namespace to filter results. Defaults to auto-detected from .swarm_workspace.",
    )
    parser.add_argument(
        "--global",
        dest="global_search",
        action="store_true",
        default=False,
        help="God-Mode: search across ALL workspaces, ignoring namespace isolation.",
    )
    args = parser.parse_args()

    db = BrainDB()

    # ── NAMESPACE RESOLUTION ────────────────────────────────────────────────
    if args.global_search:
        # God-Mode: drop the where clause entirely
        workspace_id = None
        print(f"[SEARCH] GOD-MODE — querying across all workspaces", file=sys.stderr)
    else:
        workspace_id = args.workspace or get_workspace_id()
        print(f"[SEARCH] Namespace: '{workspace_id}'", file=sys.stderr)
    # ────────────────────────────────────────────────────────────────────────

    results = db.search(
        args.query,
        workbase_id=workspace_id,  # None = global, str = namespace-scoped
        limit=args.limit,
    )

    out = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        ids = results["ids"][0]
        docs = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results.get("metadatas", [[]])[0]
        for i in range(len(ids)):
            relevance = max(0.0, 1.0 - distances[i])
            out.append(
                {
                    "id": ids[i],
                    "content": docs[i],
                    "relevance_score": round(relevance, 4),
                    "workspace": metadatas[i].get("workspace", "unknown")
                    if metadatas
                    else "unknown",
                }
            )

    print(json.dumps(out))


if __name__ == "__main__":
    main()
