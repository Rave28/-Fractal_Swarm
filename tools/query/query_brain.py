import argparse
import os
import sys

import chromadb
from chromadb.utils import embedding_functions


def main() -> None:
    # Force UTF-8 encoding for Windows console
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    # 1. Setup the CLI parser
    parser = argparse.ArgumentParser(
        description="Interrogate the Sovereign Oracle (MyBrain)"
    )
    parser.add_argument(
        "query", type=str, help="The semantic query to search your local vector memory"
    )
    parser.add_argument(
        "--results", type=int, default=3, help="Number of chunks to retrieve"
    )
    args = parser.parse_args()

    # 2. Connect to the local vector node
    # Portable path discovery
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SWARM_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

    db_path = os.path.join(SWARM_ROOT, "core", "brain", "myBrAIn", "chroma_db")
    if not os.path.exists(db_path):
        db_path = os.path.expanduser("~/mybrain_data")

    collection_name = "knowledge_broker_collection"

    print(f"Connecting to MyBrain node at {db_path}...")

    try:
        client = chromadb.PersistentClient(path=db_path)
        # Using the exact same embedding model to ensure semantic alignment
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = client.get_collection(
            name=collection_name, embedding_function=emb_fn
        )
    except Exception as e:
        print(f"Connection failed. Ensure the broker has ingested data. Error: {e}")
        sys.exit(1)

    print(f"Executing semantic search for: '{args.query}'\n")

    # 3. Execute the Vector Search
    results = collection.query(query_texts=[args.query], n_results=args.results)

    if not results["documents"] or not results["documents"][0]:
        print("No relevant memories found in the local node.")
        return

    # 4. Stream the results to the terminal
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        print(
            f"\033[96m--- Memory Chunk {i + 1} (Semantic Distance: {dist:.4f}) ---\033[0m"
        )
        print(f"\033[93mOriginal Broker Task:\033[0m {meta.get('query', 'Unknown')}")
        print(f"\n{doc}\n")


if __name__ == "__main__":
    main()
