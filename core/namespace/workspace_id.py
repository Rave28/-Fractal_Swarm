"""
workspace_id.py — Swarm Workspace Namespace Resolver
=====================================================
Single source of truth for resolving the current workspace_id from:
  1. A .swarm_workspace file in the current or any parent directory (project scope)
  2. The SWARM_WORKSPACE env var (CI / override)
  3. A slugified version of the current directory name (fallback)

Usage:
    from workspace_id import get_workspace_id, set_workspace_id

    wid = get_workspace_id()           # auto-detected
    wid = get_workspace_id("/path")    # explicit cwd override
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path


WORKSPACE_FILE = ".swarm_workspace"
ENV_VAR = "SWARM_WORKSPACE"


def _slugify(name: str) -> str:
    """Convert a directory name into a safe, lowercase workspace slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9_-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "default_workspace"


@lru_cache(maxsize=128)
def _find_workspace_file(start: str) -> str | None:
    """Walk up the directory tree looking for a .swarm_workspace file.

    Results are cached (LRU-128) to eliminate repeated filesystem I/O in
    hot OODA loops. The cache is automatically busted by set_workspace_id().
    """
    current = Path(start).resolve()
    while True:
        candidate = current / WORKSPACE_FILE
        if candidate.is_file():
            return str(candidate)
        parent = current.parent
        if parent == current:
            return None
        current = parent


def get_workspace_id(cwd: str | Path | None = None) -> str:
    """
    Resolve the active workspace_id using the priority chain:
      1. SWARM_WORKSPACE environment variable (highest priority — CI / override)
      2. .swarm_workspace file found in cwd or any parent directory
      3. Slugified version of cwd's directory name (safe fallback)

    Args:
        cwd: Directory to start the search from. Defaults to os.getcwd().

    Returns:
        A non-empty workspace_id string safe for ChromaDB metadata filtering.
    """
    # 1. Environment variable override
    env_val = os.environ.get(ENV_VAR, "").strip()
    if env_val:
        return _slugify(env_val)

    # 2. .swarm_workspace file (LRU-cached filesystem walk)
    start = Path(cwd).resolve() if cwd else Path.cwd()
    workspace_file_path = _find_workspace_file(str(start))
    if workspace_file_path:
        try:
            data = json.loads(Path(workspace_file_path).read_text(encoding="utf-8"))
            wid = str(data.get("workspace_id", "")).strip()
            if wid:
                return wid
        except (json.JSONDecodeError, OSError):
            pass

    # 3. Slugify-current-dir fallback
    return _slugify(start.name)


def set_workspace_id(workspace_id: str, target_dir: str | Path | None = None) -> Path:
    """
    Write a .swarm_workspace file into target_dir (defaults to cwd).
    Busts the LRU cache so the new file is picked up immediately.
    """
    if not workspace_id or not workspace_id.strip():
        raise ValueError("workspace_id cannot be empty.")

    target = Path(target_dir).resolve() if target_dir else Path.cwd()
    target.mkdir(parents=True, exist_ok=True)

    data = {"workspace_id": workspace_id, "created_at": _iso_now()}
    out_path = target / WORKSPACE_FILE
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── Bust the LRU cache so the new file is visible immediately ─────────
    _find_workspace_file.cache_clear()

    return out_path


def _iso_now() -> str:
    import datetime

    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        # Called as: python workspace_id.py /path/to/project
        wid = get_workspace_id(sys.argv[1])
    else:
        wid = get_workspace_id()

    print(wid)
