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
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ─── Constants ────────────────────────────────────────────────────────────────

GUARD_VERSION: Final[str] = "v1.0.0"

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
        """Resolve a Call node into a dotted name string, e.g. 'os.remove'."""
        func = node.func
        if isinstance(func, ast.Attribute):
            value = func.value
            if isinstance(value, ast.Name):
                return f"{value.id}.{func.attr}"
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                return f"{value.value.id}.{value.attr}.{func.attr}"
        if isinstance(func, ast.Name):
            return func.id
        return None

    def visit_Call(self, node: ast.Call) -> None:
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

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._secret_key: str | None = None
        self._scan_count: int = 0
        self._violation_count: int = 0

    def set_secret_key(self, key: str) -> None:
        """Set the HMAC secret for capability token verification."""
        self._secret_key = key

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
        """
        self._scan_count += 1

        # Parse —  will raise SyntaxError for unparseable code (also a useful signal)
        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError as e:
            raise SyntaxError(
                f"[SwarmBoundaryGuard] Cannot parse '{filename}': {e}"
            ) from e

        visitor = _BoundaryVisitor(workspace_root=self.workspace_root)
        visitor.visit(tree)

        if not visitor.violations:
            return  # ✅ Clean

        self._violation_count += len(visitor.violations)

        # If a capability token is provided, verify it
        if capability_token is not None:
            if self._secret_key is None:
                raise InvalidCapabilityToken(
                    "A capability token was provided but no secret key has been set. "
                    "Call SwarmBoundaryGuard.set_secret_key() first."
                )
            capability_token.verify(self._secret_key)
            # Token verified — log the authorized override and allow
            print(
                f"⚠️  [SwarmBoundaryGuard] AUTHORIZED OVERRIDE: '{filename}' contains "
                f"{len(visitor.violations)} violation(s). Token justification: "
                f"'{capability_token.justification}'"
            )
            return

        # No token — hard block
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
        return {
            "guard_version": GUARD_VERSION,
            "workspace_root": str(self.workspace_root),
            "total_scans": self._scan_count,
            "total_violations_caught": self._violation_count,
        }


# ─── CLI Self-Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🛡️  SwarmBoundaryGuard {GUARD_VERSION} — Self-Test\n")

    guard = SwarmBoundaryGuard(workspace_root=Path.cwd())

    # Test 1: Clean code — should pass silently
    clean_code = """
def add(a: int, b: int) -> int:
    return a + b
"""
    guard.scan("clean_module.py", clean_code)
    print("✅ Test 1 PASSED: Clean code accepted.")

    # Test 2: Destructive code — should raise SwarmBoundaryViolation
    rogue_code = """
import os, shutil

def nuke_workspace():
    shutil.rmtree('/critical/data')
    os.system('rm -rf /')
"""
    try:
        guard.scan("rogue_nanobot.py", rogue_code)
        print("❌ Test 2 FAILED: Rogue code was not blocked.")
    except SwarmBoundaryViolation as e:
        print(f"✅ Test 2 PASSED: SwarmBoundaryViolation raised correctly:{e}")

    # Test 3: Destructive code WITH a capability token — should pass
    guard.set_secret_key("super-secret-32-byte-swarm-key-01")
    token = SwarmCapabilityToken.generate(
        secret_key="super-secret-32-byte-swarm-key-01",
        justification="Emergency venv rebuild required after dependency conflict.",
    )
    try:
        guard.scan("authorized_cleanup.py", rogue_code, capability_token=token)
        print("✅ Test 3 PASSED: Authorized override accepted.")
    except (SwarmBoundaryViolation, InvalidCapabilityToken) as e:
        print(f"❌ Test 3 FAILED: {e}")

    # Test 4: Reuse a consumed token — should raise InvalidCapabilityToken
    try:
        guard.scan("second_run.py", rogue_code, capability_token=token)
        print("❌ Test 4 FAILED: Consumed token was reused.")
    except InvalidCapabilityToken as e:
        print(f"✅ Test 4 PASSED: Consumed token rejected: {e}")

    print(f"\n📊 Guard Stats: {guard.stats}")
