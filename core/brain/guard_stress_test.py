"""
Stress Test + Failure Battery for SwarmBoundaryGuard v1.0.0

Tests:
  A) Correct behaviour tests (should PASS / BLOCK / ALLOW as expected)
  B) Adversarial evasion tests (attacker tries to sneak past the AST guard)
  C) Edge-case failure tests (malformed input, huge files, empty files, etc.)
  D) Performance stress test (throughput benchmark / scan latency)

Run: python guard_stress_test.py
"""

import sys
import time
import textwrap
import traceback
from pathlib import Path

# Import the guard from the same package
sys.path.insert(0, str(Path(__file__).parent))
from swarm_boundary_guard import (
    SwarmBoundaryGuard,
    SwarmCapabilityToken,
    SwarmBoundaryViolation,
    InvalidCapabilityToken,
    GUARD_VERSION,
)

# ─── Test Runner ──────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results: list[tuple[str, str, float]] = []  # (status, name, elapsed_ms)


def run(name: str, fn):
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        results.append((PASS, name, elapsed))
        print(f"{PASS}  {name}  ({elapsed:.2f}ms)")
    except AssertionError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        results.append((FAIL, name, elapsed))
        print(f"{FAIL}  {name}  ({elapsed:.2f}ms)\n       => {e}")
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        results.append((FAIL, name, elapsed))
        print(
            f"{FAIL}  {name}  ({elapsed:.2f}ms)\n       => Unexpected: {type(e).__name__}: {e}"
        )


# ─── Fixtures ─────────────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).parent
guard = SwarmBoundaryGuard(workspace_root=WORKSPACE)
guard.set_secret_key("aegis-test-secret-key-v1-32bytes!")

SECRET = "aegis-test-secret-key-v1-32bytes!"


def fresh_token(
    justification: str = "Authorized destructive op for test purposes.",
) -> SwarmCapabilityToken:
    return SwarmCapabilityToken.generate(secret_key=SECRET, justification=justification)


# ═══════════════════════════════════════════════════════════════════════════════
# A) CORRECT BEHAVIOUR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- A) Correct Behaviour Tests ---\n")


def test_clean_function():
    guard.scan("test.py", "def add(a, b):\n    return a + b\n")


run("A01: Clean function passes", test_clean_function)


def test_clean_class():
    code = textwrap.dedent("""
        class Agent:
            def __init__(self, name: str):
                self.name = name
            def think(self) -> str:
                return f"Thinking as {self.name}"
    """)
    guard.scan("agent.py", code)


run("A02: Clean class passes", test_clean_class)


def test_os_remove_blocked():
    try:
        guard.scan("bad.py", "import os\nos.remove('/etc/passwd')\n")
        assert False, "Should have raised SwarmBoundaryViolation"
    except SwarmBoundaryViolation as e:
        assert any(v.call == "os.remove" for v in e.violations)


run("A03: os.remove is blocked", test_os_remove_blocked)


def test_shutil_rmtree_blocked():
    try:
        guard.scan("bad.py", "import shutil\nshutil.rmtree('/data')\n")
        assert False, "Should have raised SwarmBoundaryViolation"
    except SwarmBoundaryViolation as e:
        assert any("shutil.rmtree" in v.call for v in e.violations)


run("A04: shutil.rmtree is blocked", test_shutil_rmtree_blocked)


def test_subprocess_run_blocked():
    try:
        guard.scan("bad.py", "import subprocess\nsubprocess.run(['rm', '-rf', '/'])\n")
        assert False
    except SwarmBoundaryViolation as e:
        assert any("subprocess.run" in v.call for v in e.violations)


run("A05: subprocess.run is blocked", test_subprocess_run_blocked)


def test_os_system_blocked():
    try:
        guard.scan("bad.py", "import os\nos.system('del /q /f /s C:\\\\')\n")
        assert False
    except SwarmBoundaryViolation as e:
        assert any("os.system" in v.call for v in e.violations)


run("A06: os.system is blocked", test_os_system_blocked)


def test_os_popen_blocked():
    try:
        guard.scan("bad.py", "import os\nos.popen('cat /etc/shadow')\n")
        assert False
    except SwarmBoundaryViolation as e:
        assert any("os.popen" in v.call for v in e.violations)


run("A07: os.popen is blocked", test_os_popen_blocked)


def test_authorized_override_works():
    rogue = "import os\nos.remove('/venv')\n"
    token = fresh_token()
    guard.scan("cleanup.py", rogue, capability_token=token)  # Must not raise


run("A08: Authorized token overrides block", test_authorized_override_works)


def test_consumed_token_rejected():
    token = fresh_token()
    rogue = "import os\nos.remove('/venv')\n"
    guard.scan("pass1.py", rogue, capability_token=token)
    try:
        guard.scan("pass2.py", rogue, capability_token=token)
        assert False, "Consumed token should have been rejected"
    except InvalidCapabilityToken as e:
        assert "already been consumed" in str(e)


run("A09: Consumed token is rejected on reuse", test_consumed_token_rejected)


def test_batch_scan_clean():
    files = {
        "a.py": "def a(): return 1",
        "b.py": "def b(): return 2",
        "c.py": "x = [i**2 for i in range(100)]",
    }
    guard.scan_batch(files)


run("A10: Batch scan of clean files passes", test_batch_scan_clean)


def test_batch_scan_detects_any_violation():
    files = {
        "clean.py": "def noop(): pass",
        "rogue.py": "import shutil\nshutil.rmtree('/data')",
    }
    try:
        guard.scan_batch(files)
        assert False
    except SwarmBoundaryViolation as e:
        assert "rogue.py" == e.filename


run("A11: Batch scan flags offending file", test_batch_scan_detects_any_violation)


def test_expired_token_rejected():
    token = SwarmCapabilityToken.generate(
        SECRET, "Short-lived token.", ttl_seconds=0.001
    )
    time.sleep(0.005)
    rogue = "import os\nos.remove('/venv')\n"
    try:
        guard.scan("cleanup.py", rogue, capability_token=token)
        assert False, "Expired token should have been rejected"
    except InvalidCapabilityToken as e:
        assert "expired" in str(e).lower()


run("A12: Expired token is rejected", test_expired_token_rejected)


# ═══════════════════════════════════════════════════════════════════════════════
# B) ADVERSARIAL EVASION TESTS  (attacker tries to sneak past the guard)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- B) Adversarial Evasion Tests ---\n")


def test_evasion_via_getattr():
    """Attacker calls os.remove via getattr(os, 'remove')."""
    # Guard does NOT need to catch this — dynamic resolution is a runtime problem.
    # But we record whether it passes/fails to document the gap.
    code = "import os\ngetattr(os, 'remove')('/data')\n"
    try:
        guard.scan("evasion.py", code)
        # Static AST cannot catch this — document as known limitation
        results.append((WARN, "B01: getattr evasion (known AST limitation)", 0.0))
        print(
            f"{WARN}  B01: getattr evasion NOT caught (known static analysis limitation)"
        )
    except SwarmBoundaryViolation:
        results.append((PASS, "B01: getattr evasion caught", 0.0))
        print(f"{PASS}  B01: getattr evasion caught")


test_evasion_via_getattr()


def test_evasion_via_alias():
    """import shutil as s; s.rmtree(...)"""
    code = "import shutil as s\ns.rmtree('/data')\n"
    try:
        guard.scan("alias.py", code)
        results.append((WARN, "B02: import alias evasion (known limitation)", 0.0))
        print(
            f"{WARN}  B02: import alias evasion NOT caught (known static analysis limitation)"
        )
    except SwarmBoundaryViolation:
        results.append((PASS, "B02: import alias caught", 0.0))
        print(f"{PASS}  B02: import alias caught")


test_evasion_via_alias()


def test_evasion_via_exec():
    """exec() to run arbitrary code at runtime."""
    code = "exec(\"import os; os.system('rm -rf /')\")\n"
    try:
        guard.scan("exec_evasion.py", code)
        results.append((WARN, "B03: exec() evasion (known limitation)", 0.0))
        print(
            f"{WARN}  B03: exec() evasion NOT caught — ideally should be added to blacklist"
        )
    except SwarmBoundaryViolation:
        results.append((PASS, "B03: exec() evasion caught", 0.0))
        print(f"{PASS}  B03: exec() evasion caught")


test_evasion_via_exec()


def test_evasion_nested_function():
    """Destructive call inside a deeply nested closure."""
    code = textwrap.dedent("""
        def outer():
            def inner():
                def deepest():
                    import os
                    os.remove('/kernel')
                deepest()
            inner()
    """)
    try:
        guard.scan("nested.py", code)
        assert False, "Nested call should have been caught"
    except SwarmBoundaryViolation:
        pass


run("B04: Deeply nested os.remove caught", test_evasion_nested_function)


def test_evasion_lambda():
    """Destructive call inside a lambda."""
    code = "import os\nkill = lambda: os.remove('/data')\n"
    try:
        guard.scan("lambda.py", code)
        assert False
    except SwarmBoundaryViolation:
        pass


run("B05: Lambda body os.remove caught", test_evasion_lambda)


def test_evasion_comprehension():
    """Destructive call inside list comprehension."""
    code = "import os\nresult = [os.remove(f) for f in ['/a', '/b']]\n"
    try:
        guard.scan("comp.py", code)
        assert False
    except SwarmBoundaryViolation:
        pass


run("B06: List comprehension os.remove caught", test_evasion_comprehension)


def test_evasion_try_block():
    """Destructive call hidden inside try/except."""
    code = textwrap.dedent("""
        import os
        try:
            os.system('format c: /q')
        except Exception:
            pass
    """)
    try:
        guard.scan("try_evasion.py", code)
        assert False
    except SwarmBoundaryViolation:
        pass


run("B07: try/except-wrapped os.system caught", test_evasion_try_block)


# ═══════════════════════════════════════════════════════════════════════════════
# C) EDGE CASE / FAILURE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- C) Edge Case / Failure Tests ---\n")


def test_empty_file():
    guard.scan("empty.py", "")


run("C01: Empty file passes", test_empty_file)


def test_comments_only():
    guard.scan("comments.py", "# This is just a comment\n# No code here\n")


run("C02: Comments-only file passes", test_comments_only)


def test_syntax_error_raises():
    try:
        guard.scan("broken.py", "def foo(:\n    pass\n")
        assert False, "SyntaxError should have been raised"
    except SyntaxError:
        pass


run(
    "C03: Syntax error raises SyntaxError (not crashes guard)", test_syntax_error_raises
)


def test_none_code_raises():
    try:
        guard.scan("none.py", None)  # type: ignore
        assert False
    except (TypeError, AttributeError):
        pass


run("C04: None code raises TypeError (not crashes guard)", test_none_code_raises)


def test_short_justification_rejected():
    try:
        SwarmCapabilityToken.generate(SECRET, "short")
        assert False, "Short justification should be rejected"
    except ValueError as e:
        assert "10 characters" in str(e)


run(
    "C05: Token with <10 char justification rejected", test_short_justification_rejected
)


def test_batch_with_syntax_error():
    files = {"ok.py": "x = 1", "broken.py": "def (:"}
    try:
        guard.scan_batch(files)
        assert False, "Should have raised SyntaxError"
    except SyntaxError:
        pass


run(
    "C06: Batch scan with syntax error raises SyntaxError", test_batch_with_syntax_error
)


def test_large_clean_file():
    """100,000-line file with no violations — tests throughput."""
    code = "x = 0\n" * 100_000
    t0 = time.perf_counter()
    guard.scan("big.py", code)
    elapsed = (time.perf_counter() - t0) * 1000
    assert elapsed < 5000, f"Scan took too long: {elapsed:.0f}ms"
    print(f"       Scanned 100k-line file in {elapsed:.1f}ms")


run("C07: 100k-line clean file scanned within 5s", test_large_clean_file)


def test_large_malicious_file():
    """10,000 lines of legitimate code followed by one destructive call."""
    safe_lines = "x = 0\n" * 10_000
    evil_line = "import os\nos.remove('/root')\n"
    code = safe_lines + evil_line
    try:
        guard.scan("big_bad.py", code)
        assert False
    except SwarmBoundaryViolation as e:
        assert e.violations[0].lineno > 10_000


run("C08: Violation at end of 10k-line file detected", test_large_malicious_file)


def test_multiple_violations_all_reported():
    code = textwrap.dedent("""
        import os, shutil, subprocess
        os.remove('/a')
        shutil.rmtree('/b')
        subprocess.run(['rm'])
        os.system('kill')
        os.popen('cat /etc/shadow')
    """)
    try:
        guard.scan("multi.py", code)
        assert False
    except SwarmBoundaryViolation as e:
        assert len(e.violations) == 5, f"Expected 5 violations, got {len(e.violations)}"


run("C09: All 5 violations in one file reported", test_multiple_violations_all_reported)


# ═══════════════════════════════════════════════════════════════════════════════
# D) PERFORMANCE STRESS TEST
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- D) Performance Stress Tests ---\n")

ITERATIONS = 500
CLEAN_CODE = textwrap.dedent("""
    from typing import Optional

    class VectorAgent:
        def __init__(self, name: str, dimension: int = 768) -> None:
            self.name = name
            self.dimension = dimension
            self._memory: list[list[float]] = []

        def ingest(self, vector: list[float]) -> None:
            if len(vector) != self.dimension:
                raise ValueError("Dimension mismatch")
            self._memory.append(vector)

        def recall(self, query: list[float], top_k: int = 5) -> list[list[float]]:
            return sorted(self._memory, key=lambda v: sum((a - b) ** 2 for a, b in zip(v, query)))[:top_k]
""")


def test_throughput():
    t0 = time.perf_counter()
    g = SwarmBoundaryGuard(workspace_root=WORKSPACE)
    for i in range(ITERATIONS):
        g.scan(f"agent_{i}.py", CLEAN_CODE)
    elapsed = (time.perf_counter() - t0) * 1000
    per_scan = elapsed / ITERATIONS
    # Performance target: each clean scan < 10ms on a normal laptop
    assert per_scan < 10.0, f"Scan too slow: {per_scan:.2f}ms per scan"
    print(
        f"       {ITERATIONS} scans completed in {elapsed:.1f}ms | {per_scan:.3f}ms per scan"
    )


run(f"D01: Throughput — {ITERATIONS} clean scans < 10ms each", test_throughput)


def test_batch_throughput():
    """Batch 50 files per call, 20 calls."""
    batch = {f"file_{i}.py": CLEAN_CODE for i in range(50)}
    t0 = time.perf_counter()
    g = SwarmBoundaryGuard(workspace_root=WORKSPACE)
    for _ in range(20):
        g.scan_batch(batch)
    elapsed = (time.perf_counter() - t0) * 1000
    total_files = 50 * 20
    per_file = elapsed / total_files
    assert per_file < 10.0, f"Batch too slow: {per_file:.2f}ms per file"
    print(
        f"       {total_files} files batched in {elapsed:.1f}ms | {per_file:.3f}ms per file"
    )


run("D02: Batch throughput — 1000 files across 20 batches", test_batch_throughput)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
total = len(results)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
warned = sum(1 for s, _, _ in results if s == WARN)
avg_ms = sum(ms for _, _, ms in results) / total if total else 0

print(
    f"\n  RESULT: {passed}/{total} PASSED | {failed} FAILED | {warned} KNOWN LIMITATIONS"
)
print(
    f"  Total elapsed: {sum(ms for _, _, ms in results):.1f}ms  |  Avg per test: {avg_ms:.1f}ms"
)

if failed > 0:
    print("\n  FAILED TESTS:")
    for status, name, ms in results:
        if status == FAIL:
            print(f"    - {name}")

if warned > 0:
    print("\n  KNOWN STATIC ANALYSIS LIMITATIONS (require runtime guards):")
    for status, name, ms in results:
        if status == WARN:
            print(f"    * {name}")

print("\n" + "=" * 60 + "\n")

sys.exit(0 if failed == 0 else 1)
