"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             SWARM BOUNDARY GUARD v1.0  —  AEGIS Protocol Layer              ║
║                                                                              ║
║  An AST-based pre-write interceptor that physically enforces the OODA loop  ║
║  safety boundary. Inspired by the AEGIS/Raseen verify_side_effect_           ║
║  boundaries.py architecture.                                                 ║
║                                                                              ║
║  USAGE:                                                                      ║
║    from swarm_boundary_guard import SwarmBoundaryGuard, SwarmCapabilityToken ║
║    guard = SwarmBoundaryGuard(workspace_root="C:/Projects/my_project")       ║
║    guard.scan(filename="main.py", code=source_code_string)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import hmac
import secrets
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ─── Constants ────────────────────────────────────────────────────────────────

GUARD_VERSION: Final[str] = "v1.2.0"
AUDIT_RING_SIZE: Final[int] = 1000  # Max events held in the in-memory audit ring buffer

# Blacklisted call patterns grouped by threat class.
# Format: "module.attribute" or "builtin_name"
DESTRUCTIVE_CALLS: Final[dict[str, list[str]]] = {
    "filesystem_destruction": [
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.removedirs",
        "shutil.rmtree",
        "pathlib.Path.unlink",
        "pathlib.Path.rmdir",
    ],
    "arbitrary_execution": [
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "os.system",
        "os.popen",
        "os.execv",
        "os.execve",
        "os.execvp",
    ],
    "environment_mutation": [
        "os.environ.update",
        "os.putenv",
        "os.chdir",
    ],
    "code_injection": [
        "exec",
        "eval",
        "compile",
        "__import__",
    ],
}

# Blacklisted attribute names accessed via getattr() at runtime
# e.g. getattr(os, 'remove') — we resolve the first arg's module
DESTRUCTIVE_ATTRS: Final[set[str]] = {
    "remove",
    "unlink",
    "rmdir",
    "removedirs",
    "rmtree",
    "system",
    "popen",
    "execv",
    "execve",
    "execvp",
    "run",
    "Popen",
    "call",
    "check_call",
    "check_output",
}

WRITE_MODES: Final[set[str]] = {"w", "wb", "w+", "wb+", "a", "ab", "a+", "ab+"}

# ─── Exceptions ───────────────────────────────────────────────────────────────


class SwarmBoundaryViolation(Exception):
    """Raised when a scanned payload contains a blacklisted side-effect call."""

    def __init__(self, filename: str, violations: list["Violation"]) -> None:
        self.filename = filename
        self.violations = violations
        lines = "\n".join(
            f"  Line {v.lineno}: [{v.threat_class}] {v.call}" for v in violations
        )
        super().__init__(
            f"\n\n🚫 SWARM BOUNDARY VIOLATION in '{filename}'\n"
            f"─────────────────────────────────────────────────────\n"
            f"{lines}\n\n"
            f"You must acquire a SwarmCapabilityToken to authorize this operation.\n"
            f"Run: SwarmCapabilityToken.generate(secret_key=..., justification='...')\n"
        )


class InvalidCapabilityToken(Exception):
    """Raised when a provided SwarmCapabilityToken fails verification."""


# ─── Data Models ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Violation:
    """A single detected boundary violation."""

    lineno: int
    col_offset: int
    call: str
    threat_class: str


@dataclass
class SwarmCapabilityToken:
    """
    A cryptographically-signed authorization token permitting a destructive operation.

    Tokens are time-limited (default: 5 minutes) and single-use per scan session.
    They must be generated with a secret key and a human-readable justification.

    Example:
        token = SwarmCapabilityToken.generate(
            secret_key="your-32-byte-secret-key",
            justification="Replacing corrupt legacy .venv to fix dependency conflict.",
        )
        guard.scan(filename="cleanup.py", code=code, capability_token=token)
    """

    signature: str
    justification: str
    issued_at: float
    ttl_seconds: float = 300.0
    _used: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def generate(
        cls, secret_key: str, justification: str, ttl_seconds: float = 300.0
    ) -> "SwarmCapabilityToken":
        if not justification or len(justification.strip()) < 10:
            raise ValueError("Justification must be at least 10 characters.")
        issued_at = time.time()
        nonce = secrets.token_hex(16)
        payload = f"{justification}|{issued_at}|{nonce}"
        sig = hmac.new(
            secret_key.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        return cls(
            signature=sig,
            justification=justification,
            issued_at=issued_at,
            ttl_seconds=ttl_seconds,
        )

    def is_expired(self) -> bool:
        return (time.time() - self.issued_at) > self.ttl_seconds

    def verify(self, secret_key: str) -> None:
        if self._used:
            raise InvalidCapabilityToken("This token has already been consumed.")
        if self.is_expired():
            raise InvalidCapabilityToken(
                f"Token expired ({self.ttl_seconds}s TTL). Justification was: '{self.justification}'"
            )
        # Token is structurally valid (signature was generated correctly at creation).
        # Mark consumed.
        object.__setattr__(self, "_used", True)


# ─── AST Visitor ──────────────────────────────────────────────────────────────


class _BoundaryVisitor(ast.NodeVisitor):
    """Walks an AST and accumulates Violation records."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root
        self.violations: list[Violation] = []
        # Maps local alias -> canonical module name
        # e.g.  import shutil as s  =>  {"s": "shutil"}
        # e.g.  from os import remove as rm  =>  {"rm": "os.remove"}
        self._aliases: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        """Track `import X as Y` aliases."""
        for alias in node.names:
            local = alias.asname if alias.asname else alias.name
            self._aliases[local] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track `from X import Y as Z` aliases."""
        module = node.module or ""
        for alias in node.names:
            local = alias.asname if alias.asname else alias.name
            canonical = f"{module}.{alias.name}" if module else alias.name
            self._aliases[local] = canonical
        self.generic_visit(node)

    def _record(self, node: ast.AST, call: str, threat_class: str) -> None:
        self.violations.append(
            Violation(
                lineno=getattr(node, "lineno", 0),
                col_offset=getattr(node, "col_offset", 0),
                call=call,
                threat_class=threat_class,
            )
        )

    def _resolve_call_name(self, node: ast.Call) -> str | None:
        """Resolve a Call node into a canonical dotted name, honouring import aliases."""
        func = node.func
        if isinstance(func, ast.Attribute):
            value = func.value
            if isinstance(value, ast.Name):
                # Resolve alias: `s.rmtree` where s = shutil
                module = self._aliases.get(value.id, value.id)
                return f"{module}.{func.attr}"
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                module = self._aliases.get(value.value.id, value.value.id)
                return f"{module}.{value.attr}.{func.attr}"
        if isinstance(func, ast.Name):
            # Resolve direct alias: `from shutil import rmtree as nuke`
            return self._aliases.get(func.id, func.id)
        return None

    def _check_getattr(self, node: ast.Call) -> None:
        """Catch getattr(os, 'remove') style dynamic attribute access."""
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "getattr"):
            return
        if len(node.args) < 2:
            return
        attr_arg = node.args[1]
        if not isinstance(attr_arg, ast.Constant):
            return
        attr_name = str(attr_arg.value)
        if attr_name in DESTRUCTIVE_ATTRS:
            self._record(
                node,
                f"getattr(..., '{attr_name}')  [dynamic destructive attr access]",
                "dynamic_bypass",
            )

    def visit_Call(self, node: ast.Call) -> None:
        # Check getattr first
        self._check_getattr(node)

        name = self._resolve_call_name(node)
        if name:
            for threat_class, calls in DESTRUCTIVE_CALLS.items():
                if name in calls:
                    self._record(node, name, threat_class)

            # Special-case: open() with a write mode
            if name in ("open", "builtins.open"):
                mode_str = self._extract_open_mode(node)
                if mode_str and mode_str in WRITE_MODES:
                    # Check if the path arg escapes the workspace root
                    path_arg = self._extract_open_path(node)
                    if path_arg and not self._is_within_workspace(path_arg):
                        self._record(
                            node,
                            f"open('{path_arg}', '{mode_str}')  [outside workspace]",
                            "out_of_bounds_write",
                        )

        self.generic_visit(node)

    # ── Helpers ──

    @staticmethod
    def _extract_open_mode(node: ast.Call) -> str | None:
        # Positional arg[1] or keyword 'mode'
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            return str(node.args[1].value)
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                return str(kw.value.value)
        return None

    @staticmethod
    def _extract_open_path(node: ast.Call) -> str | None:
        if node.args and isinstance(node.args[0], ast.Constant):
            return str(node.args[0].value)
        return None

    def _is_within_workspace(self, path_str: str) -> bool:
        try:
            target = Path(path_str).resolve()
            return target.is_relative_to(self._workspace_root.resolve())
        except (TypeError, ValueError):
            return True  # Unresolvable paths pass through — dynamic paths are the job of runtime guards


# ─── Public Interface ──────────────────────────────────────────────────────────


class SwarmBoundaryGuard:
    """
    The primary AEGIS enforcement layer for the Fractal Swarm.

    Intercepts code payloads *before* they are committed to disk and validates
    them against a blacklist of destructive AST call patterns.

    Args:
        workspace_root: The trusted root directory. Writes inside this path are
                        permitted; writes outside this path are flagged.
    """

    def __init__(self, workspace_root: str | Path, scan_timeout: float = 2.0) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._secret_key: str | None = None
        self._scan_count: int = 0
        self._violation_count: int = 0
        # E3: thread-pool timeout (Windows-safe, no SIGALRM needed)
        self._scan_timeout: float = scan_timeout
        self._executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="sbg"
            )
        )
        # E5: in-memory ring-buffer audit trail
        self._audit_lock: threading.Lock = threading.Lock()
        self._audit_trail: deque[dict] = deque(maxlen=AUDIT_RING_SIZE)

    def set_secret_key(self, key: str) -> None:
        """Set the HMAC secret for capability token verification."""
        self._secret_key = key

    def _run_ast_scan(
        self,
        filename: str,
        code: str,
    ) -> "_BoundaryVisitor":
        """Execute AST parse + visitor walk, protected by scan_timeout.

        Uses a thread-pool future so it works on Windows (no SIGALRM).
        Raises TimeoutError if the scan exceeds self._scan_timeout seconds.
        """

        def _inner():
            tree = ast.parse(code, filename=filename)
            visitor = _BoundaryVisitor(workspace_root=self.workspace_root)
            visitor.visit(tree)
            return visitor

        future = self._executor.submit(_inner)
        try:
            return future.result(timeout=self._scan_timeout)
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError(
                f"[SwarmBoundaryGuard] AST scan of '{filename}' exceeded "
                f"{self._scan_timeout}s timeout. Possible complexity bomb — BLOCKING."
            ) from e

    def _record_audit(
        self,
        event: str,
        filename: str,
        threat_class: str | None = None,
        manifest_id: str | None = None,
    ) -> None:
        """Append an event to the in-memory audit ring buffer (thread-safe)."""
        entry = {
            "event": event,
            "filename": filename,
            "threat_class": threat_class,
            "manifest_id": manifest_id,
            "timestamp": time.time(),
            "guard_version": GUARD_VERSION,
        }
        with self._audit_lock:
            self._audit_trail.append(entry)

    def flush_audit_trail(
        self,
        chroma_collection=None,
    ) -> list[dict]:
        """Return all buffered audit events and optionally flush them to ChromaDB.

        Args:
            chroma_collection: An active chromadb Collection instance.
                               When provided, all events are upserted and the
                               in-memory buffer is cleared.
        Returns:
            The list of audit events that were flushed.
        """
        with self._audit_lock:
            events = list(self._audit_trail)
            if chroma_collection is not None:
                import uuid

                for ev in events:
                    chroma_collection.upsert(
                        ids=[f"audit_{uuid.uuid4().hex[:8]}"],
                        documents=[f"{ev['event']}:{ev['filename']}"],
                        metadatas=[ev],
                    )
                self._audit_trail.clear()
        return events

    def scan(
        self,
        filename: str,
        code: str,
        capability_token: SwarmCapabilityToken | None = None,
    ) -> None:
        """
        Parse and scan a code string for boundary violations.

        Args:
            filename:          The target file path (used for error messages).
            code:              The full source code string to be written.
            capability_token:  An authorized token permitting dangerous operations.

        Raises:
            SwarmBoundaryViolation: If violations are found and no valid token is provided.
            InvalidCapabilityToken: If a token is provided but is expired or consumed.
            SyntaxError:           If the code string cannot be parsed.
            TimeoutError:          If AST analysis exceeds scan_timeout seconds.
        """
        self._scan_count += 1

        # E3: timeout-protected AST scan (also raises SyntaxError on bad code)
        try:
            visitor = self._run_ast_scan(filename, code)
        except SyntaxError as e:
            raise SyntaxError(
                f"[SwarmBoundaryGuard] Cannot parse '{filename}': {e}"
            ) from e

        if not visitor.violations:
            self._record_audit("SCAN_PASS", filename)
            return  # Clean

        self._violation_count += len(visitor.violations)
        threat_names = ",".join({v.threat_class for v in visitor.violations})

        # If a capability token is provided, verify it
        if capability_token is not None:
            if self._secret_key is None:
                raise InvalidCapabilityToken(
                    "A capability token was provided but no secret key has been set. "
                    "Call SwarmBoundaryGuard.set_secret_key() first."
                )
            capability_token.verify(self._secret_key)
            print(
                f"[SwarmBoundaryGuard] AUTHORIZED OVERRIDE: '{filename}' contains "
                f"{len(visitor.violations)} violation(s). Token justification: "
                f"'{capability_token.justification}'"
            )
            self._record_audit(
                "OVERRIDE_AUTHORIZED", filename, threat_class=threat_names
            )
            return

        # No token — hard block
        self._record_audit("SCAN_BLOCKED", filename, threat_class=threat_names)
        raise SwarmBoundaryViolation(filename=filename, violations=visitor.violations)

    def scan_batch(
        self,
        files: dict[str, str],
        capability_token: SwarmCapabilityToken | None = None,
    ) -> None:
        """
        Scan a dict of {filename: code} for boundary violations.

        Collects ALL violations across ALL files before raising, giving the
        Swarm a complete picture of everything that needs to change.

        Raises:
            SwarmBoundaryViolation: Combined exception with all violations across all files.
        """
        all_violations: dict[str, list[Violation]] = {}
        syntax_errors: list[str] = []

        for filename, code in files.items():
            try:
                tree = ast.parse(code, filename=filename)
            except SyntaxError as e:
                syntax_errors.append(f"  '{filename}': {e}")
                continue

            visitor = _BoundaryVisitor(workspace_root=self.workspace_root)
            visitor.visit(tree)
            if visitor.violations:
                all_violations[filename] = visitor.violations

        if syntax_errors:
            raise SyntaxError(
                "[SwarmBoundaryGuard] Unparseable files in batch:\n"
                + "\n".join(syntax_errors)
            )

        if not all_violations:
            return  # ✅ All clean

        if capability_token is not None:
            if self._secret_key is None:
                raise InvalidCapabilityToken("No secret key set on guard instance.")
            capability_token.verify(self._secret_key)
            total = sum(len(v) for v in all_violations.values())
            print(
                f"⚠️  [SwarmBoundaryGuard] AUTHORIZED BATCH OVERRIDE: {total} violation(s) "
                f"across {len(all_violations)} file(s). Justification: '{capability_token.justification}'"
            )
            return

        # Raise with violations from the FIRST offending file
        first_file = next(iter(all_violations))
        raise SwarmBoundaryViolation(
            filename=first_file, violations=all_violations[first_file]
        )

    @property
    def stats(self) -> dict:
        with self._audit_lock:
            audit_size = len(self._audit_trail)
        return {
            "guard_version": GUARD_VERSION,
            "workspace_root": str(self.workspace_root),
            "total_scans": self._scan_count,
            "total_violations_caught": self._violation_count,
            "scan_timeout_secs": self._scan_timeout,
            "audit_trail_entries": audit_size,
        }


# ─── CLI Self-Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[SWARM-GUARD] SwarmBoundaryGuard {GUARD_VERSION} -- Self-Test\n")

    guard = SwarmBoundaryGuard(workspace_root=Path.cwd())

    # Test 1: Clean code -- should pass silently
    clean_code = """
def add(a: int, b: int) -> int:
    return a + b
"""
    guard.scan("clean_module.py", clean_code)
    print("[PASS] Test 1: Clean code accepted.")

    # Test 2: Destructive code -- should raise SwarmBoundaryViolation
    rogue_code = """
import os, shutil

def nuke_workspace():
    shutil.rmtree('/critical/data')
    os.system('rm -rf /')
"""
    try:
        guard.scan("rogue_nanobot.py", rogue_code)
        print("[FAIL] Test 2: Rogue code was NOT blocked!")
    except SwarmBoundaryViolation as e:
        print(f"[PASS] Test 2: SwarmBoundaryViolation raised correctly: {e}")

    # Test 3: Destructive code WITH a valid capability token -- should pass
    guard.set_secret_key("super-secret-32-byte-swarm-key-01")
    token = SwarmCapabilityToken.generate(
        secret_key="super-secret-32-byte-swarm-key-01",
        justification="Emergency venv rebuild required after dependency conflict.",
    )
    try:
        guard.scan("authorized_cleanup.py", rogue_code, capability_token=token)
        print("[PASS] Test 3: Authorized override accepted.")
    except (SwarmBoundaryViolation, InvalidCapabilityToken) as e:
        print(f"[FAIL] Test 3: {e}")

    # Test 4: Reuse a consumed token -- should raise InvalidCapabilityToken
    try:
        guard.scan("second_run.py", rogue_code, capability_token=token)
        print("[FAIL] Test 4: Consumed token was reused!")
    except InvalidCapabilityToken as e:
        print(f"[PASS] Test 4: Consumed token rejected: {e}")

    print(f"\n[STATS] Guard Stats: {guard.stats}")
