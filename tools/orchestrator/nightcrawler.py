#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         NIGHTCRAWLER DAEMON v1.0  —  Fractal Swarm Ingestion Engine         ║
║                                                                              ║
║  Autonomous web crawler that fetches, chunks, and burns documentation       ║
║  into the Oracle's ChromaDB vector store under a sealed workspace namespace. ║
║                                                                              ║
║  USAGE:                                                                      ║
║    python nightcrawler.py --target https://tokio.rs/tokio/tutorial --depth 2║
║    python nightcrawler.py --target https://docs.rs/tokio --depth 1          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin, urlparse

# ─── Path bootstrap ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent  # Fractal_Swarm root
sys.path.insert(0, str(_ROOT / "core" / "namespace"))
sys.path.insert(0, str(_ROOT / "pipelines" / "sandboxes"))
sys.path.insert(0, str(_ROOT / "core" / "brain"))

# ─── External deps ────────────────────────────────────────────────────────────
try:
    import httpx
except ImportError:
    print("[ERR] httpx not installed. Run: pip install httpx")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("[ERR] beautifulsoup4 not installed. Run: pip install beautifulsoup4 lxml")
    sys.exit(1)

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("[ERR] chromadb not installed. Run: pip install chromadb")
    sys.exit(1)

# ─── Swarm internals ──────────────────────────────────────────────────────────
try:
    from workspace_id import get_workspace_id

    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

try:
    from oracle_pure import chunk_intelligence  # force_split chunker

    _ORACLE_AVAILABLE = True
except ImportError:
    _ORACLE_AVAILABLE = False

# ─── Constants ────────────────────────────────────────────────────────────────
NIGHTCRAWLER_VERSION = "v1.0.0"
DB_PATH = os.path.expanduser("~/mybrain_data")
STATUS_FILE = Path(
    os.environ.get(
        "NIGHTCRAWLER_STATE", r"d:\Temp\vibe_backend\nightcrawler_state.json"
    )
)
CHUNK_MAX = 800
HTTP_TIMEOUT = 20.0
CRAWL_DELAY = 0.8  # polite delay between requests (seconds)
MAX_TEXT_PER_PAGE = 40_000  # cap per-page text to avoid memory issues

STRIP_TAGS = {"script", "style", "nav", "footer", "head", "meta", "noscript"}
DASHBOARD_URL = "http://127.0.0.1:8000"


# ─── Data structures ─────────────────────────────────────────────────────────


class CrawlResult(NamedTuple):
    url: str
    title: str
    text: str
    links: list[str]


class NightcrawlerStats:
    def __init__(self, target: str, workspace: str):
        self.target = target
        self.workspace = workspace
        self.start_time = time.time()
        self.pages_visited = 0
        self.chunks_ingested = 0
        self.errors = 0
        self.status = "BOOTING"

    def to_dict(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            "status": self.status,
            "target": self.target,
            "workspace": self.workspace,
            "pages_visited": self.pages_visited,
            "chunks_ingested": self.chunks_ingested,
            "errors": self.errors,
            "elapsed_seconds": round(elapsed, 1),
            "last_update": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


# ─── Core functions ───────────────────────────────────────────────────────────


def fetch_page(client: httpx.Client, url: str) -> CrawlResult | None:
    """Fetch a URL and extract clean text + outbound links."""
    try:
        resp = client.get(url, timeout=HTTP_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("content-type", ""):
            return None
    except Exception as e:
        print(f"  [FETCH ERR] {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Strip noise elements
    for tag in soup(STRIP_TAGS):
        tag.decompose()

    title = soup.find("title") or soup.find("h1") or soup.find("h2")
    title_text = (
        title.get_text(strip=True) if title else urlparse(url).path.split("/")[-1]
    )

    # Prefer main content areas
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"(content|main|body|docs?)", re.I))
        or soup.find("body")
    )
    raw_text = (main or soup).get_text(separator="\n", strip=True)
    clean_text = re.sub(r"\n{3,}", "\n\n", raw_text)[:MAX_TEXT_PER_PAGE]

    # Collect same-domain links
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"].split("#")[0].strip()
        if not href or href.startswith(("javascript:", "mailto:")):
            continue
        full = urljoin(url, href)
        if full.startswith(base):
            links.append(full)

    return CrawlResult(url=url, title=title_text, text=clean_text, links=links)


def chunk_text_safe(text: str) -> list[str]:
    """Use oracle_pure.chunk_intelligence if available, else fallback."""
    if _ORACLE_AVAILABLE:
        return chunk_intelligence(
            text, max_chars=CHUNK_MAX, overflow_strategy="force_split"
        )
    # Fallback: simple line splitter
    chunks, buf = [], ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > CHUNK_MAX and buf:
            chunks.append(buf.strip())
            buf = line + "\n"
        else:
            buf += line + "\n"
        if len(line) > CHUNK_MAX:
            while len(line) > CHUNK_MAX:
                chunks.append(line[:CHUNK_MAX])
                line = line[CHUNK_MAX:]
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c.strip()]


def write_status(stats: NightcrawlerStats) -> None:
    """Write machine-readable status file for the NightcrawlerStatus dashboard widget."""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    except Exception:
        pass


def ping_dashboard(stats: NightcrawlerStats) -> None:
    """Attempt to update the Sovereign Dashboard backend with live telemetry."""
    try:
        with httpx.Client(timeout=2.0) as c:
            c.post(f"{DASHBOARD_URL}/api/nightcrawler/status", json=stats.to_dict())
    except Exception:
        pass


def crawl_and_ingest(
    target: str,
    depth: int = 2,
    workspace: str | None = None,
    max_pages: int = 50,
    dry_run: bool = False,
) -> NightcrawlerStats:
    """Main OODA loop: crawl → chunk → ingest → report."""

    # ── Namespace ──────────────────────────────────────────────────────────────
    if workspace is None:
        if _WS_AVAILABLE:
            workspace = get_workspace_id()
        else:
            domain = urlparse(target).netloc.replace(".", "_").replace("-", "_")
            workspace = f"nightcrawler_{domain}"

    stats = NightcrawlerStats(target=target, workspace=workspace)
    print(f"\n{'=' * 72}")
    print(f"  NIGHTCRAWLER {NIGHTCRAWLER_VERSION}  —  ACTIVE HUNT")
    print(f"  Target    : {target}")
    print(f"  Namespace : {workspace}")
    print(f"  Depth     : {depth}  |  Max pages: {max_pages}")
    print(
        f"  Chunker   : {'oracle_pure (force_split)' if _ORACLE_AVAILABLE else 'fallback'}"
    )
    print(f"{'=' * 72}\n")

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    col_name = f"nightcrawler_{workspace}"
    collection = chroma_client.get_or_create_collection(
        name=col_name, embedding_function=emb_fn
    )
    print(f"[DB] Collection: '{col_name}'  existing vectors: {collection.count()}")

    # ── BFS crawler ────────────────────────────────────────────────────────────
    base_domain = f"{urlparse(target).scheme}://{urlparse(target).netloc}"
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(target, 0)])  # (url, current_depth)

    headers = {
        "User-Agent": f"FractalSwarm-Nightcrawler/{NIGHTCRAWLER_VERSION} (polite crawler)",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    stats.status = "ACTIVE HUNT"
    write_status(stats)

    with httpx.Client(headers=headers) as client:
        while queue and len(visited) < max_pages:
            url, current_depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            print(f"[CRAWL D{current_depth}] {url}")
            result = fetch_page(client, url)
            stats.pages_visited += 1

            if result is None:
                stats.errors += 1
                write_status(stats)
                continue

            # ── Chunk with force_split ─────────────────────────────────────────
            chunks = chunk_text_safe(result.text)
            print(
                f"         → '{result.title[:60]}' | {len(result.text):,} chars → {len(chunks)} chunks"
            )

            # ── Ingest chunks into ChromaDB ────────────────────────────────────
            if not dry_run:
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
                    doc_id = f"nc_{workspace}_{chunk_hash}_{i}"
                    metadata = {
                        "workspace": workspace,
                        "source_url": url,
                        "page_title": result.title[:120],
                        "chunk_index": i,
                        "depth": current_depth,
                        "ingested_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "chunker": "force_split" if _ORACLE_AVAILABLE else "fallback",
                    }
                    try:
                        collection.upsert(
                            ids=[doc_id],
                            documents=[chunk],
                            metadatas=[metadata],
                        )
                        stats.chunks_ingested += 1
                    except Exception as e:
                        print(f"         [INGEST ERR] chunk {i}: {e}")
                        stats.errors += 1

                print(
                    f"         ✅ Ingested {len(chunks)} chunks [{stats.chunks_ingested} total]"
                )

            # ── Enqueue next depth ─────────────────────────────────────────────
            if current_depth < depth:
                for link in dict.fromkeys(result.links):  # deduplicate, preserve order
                    if link not in visited and link.startswith(base_domain):
                        queue.append((link, current_depth + 1))

            write_status(stats)
            ping_dashboard(stats)
            time.sleep(CRAWL_DELAY)

    # ── Final report ──────────────────────────────────────────────────────────
    stats.status = "HUNT COMPLETE"
    write_status(stats)
    ping_dashboard(stats)

    print(f"\n{'=' * 72}")
    print(f"  HUNT COMPLETE")
    print(f"  Pages crawled  : {stats.pages_visited}")
    print(f"  Vectors burned : {stats.chunks_ingested}")
    print(f"  Errors         : {stats.errors}")
    print(f"  Collection     : '{col_name}'  total: {collection.count()}")
    print(f"  Elapsed        : {time.time() - stats.start_time:.1f}s")
    print(f"{'=' * 72}\n")

    return stats


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nightcrawler — Fractal Swarm web ingestion daemon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", required=True, help="Seed URL to crawl")
    parser.add_argument(
        "--depth", type=int, default=2, help="BFS crawl depth (0=seed only)"
    )
    parser.add_argument(
        "--workspace", default=None, help="Override namespace (default: auto-detect)"
    )
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to visit")
    parser.add_argument(
        "--dry-run", action="store_true", help="Crawl but do not ingest into ChromaDB"
    )
    args = parser.parse_args()

    crawl_and_ingest(
        target=args.target,
        depth=args.depth,
        workspace=args.workspace or "tokio_docs",
        max_pages=args.max_pages,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
