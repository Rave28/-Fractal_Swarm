import os
import re
import json
import subprocess
import sys
from pathlib import Path
import datetime


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9_-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "default_workspace"


def scaffold_swarm_target(target_dir: str):
    print(f"[*] Initializing V2 Fractal Swarm payload at: {target_dir}")
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    # Require uv for ultra-fast, clean Python 3.13 venvs
    uv_bin = r"C:\Users\Admin\AppData\Local\Programs\Jan\uv.exe"
    if not os.path.exists(uv_bin):
        print(f"[ERROR] Critical requirement missing: uv binary not found at {uv_bin}")
        sys.exit(1)

    venv_path = target_path / ".venv"
    print("[*] Bootstrapping Python 3.13 Virtual Environment...")
    subprocess.run([uv_bin, "venv", "-p", "3.13", str(venv_path)], check=True)

    print("[*] Injecting base Swarm dependencies (PyTest, AsyncIO patterns)...")
    reqs = ["pytest", "anyio"]
    subprocess.run(
        [uv_bin, "pip", "install", *reqs, "--python", str(venv_path)], check=True
    )

    # Initialize a dummy implement_code_summary.md for the Memory Commit
    summary_path = target_path / "implement_code_summary.md"
    if not summary_path.exists():
        with open(summary_path, "w") as f:
            f.write("# Memory Commit Log\n\n_V2 Swarm Origin Signature_\n")

    # --- NAMESPACE PROTOCOL: Write .swarm_workspace identity file ---
    workspace_id = _slugify(target_path.name)
    ws_file = target_path / ".swarm_workspace"
    ws_data = {
        "workspace_id": workspace_id,
        "display_name": target_path.name,
        "created_at": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "swarm_version": "v2.0",
    }
    ws_file.write_text(json.dumps(ws_data, indent=2), encoding="utf-8")
    print(f"[*] Namespace assigned: '{workspace_id}'  (.swarm_workspace written)")
    # ----------------------------------------------------------------

    print(f"[SUCCESS] V2 Swarm Target Scaffolded Successfully at {target_path}")
    print(
        "[*] [NEXT] You may now direct the DeepCode Nanobots to OBSERVE this directory."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python swarm_init.py <target_directory_path>")
        sys.exit(1)

    target = sys.argv[1]
    scaffold_swarm_target(target)
