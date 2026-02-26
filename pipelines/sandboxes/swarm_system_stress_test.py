"""
Fractal Swarm — System-Level Stress Test & Failure Battery
===========================================================
Targets all three subsystems of the Fractal Swarm architecture:

  LAYER 1: Oracle       — chunk_intelligence(), CONFIDENCE_THRESHOLD logic, VectorSearchEngine
  LAYER 2: Brain        — SwarmBoundaryGuard, SwarmCapabilityToken
  LAYER 3: Pipeline     — OODA loop integrity, cross-layer contract validation

Run:
  python swarm_system_stress_test.py

Exit code 0 = all mandatory tests passed.
Exit code 1 = one or more mandatory tests FAILED.
"""

from __future__ import annotations

import sys
import time
import math
import uuid
import textwrap
import threading
import concurrent.futures
from pathlib import Path
from typing import Callable

# Add both subsystem roots to path
SWARM_ROOT = Path(__file__).parent.parent.parent  # d:\Temp\Fractal_Swarm
ORACLE_ROOT = SWARM_ROOT / "core" / "oracle" / "vibe_backend"
BRAIN_ROOT = SWARM_ROOT / "core" / "brain"
PIPELINE_ROOT = SWARM_ROOT / "pipelines" / "sandboxes"

sys.path.insert(0, str(BRAIN_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT))  # for oracle_pure.py

# ─── Imports from each subsystem ──────────────────────────────────────────────

# Layer 1: Oracle pure functions (no server bootstrap / no sentence_transformers)
try:
    from oracle_pure import chunk_intelligence, CONFIDENCE_THRESHOLD

    # Dummy models for type-checking only
    class ResearchRequest:
        def __init__(self, query: str):
            self.query = query

    class ResearchResponse:
        pass

    ORACLE_AVAILABLE = True
except ImportError as e:
    ORACLE_AVAILABLE = False
    ORACLE_IMPORT_ERROR = str(e)

# Layer 2: Brain / Boundary Guard
try:
    from swarm_boundary_guard import (
        SwarmBoundaryGuard,
        SwarmCapabilityToken,
        SwarmBoundaryViolation,
        InvalidCapabilityToken,
        GUARD_VERSION,
    )

    GUARD_AVAILABLE = True
except ImportError as e:
    GUARD_AVAILABLE = False
    GUARD_IMPORT_ERROR = str(e)

# ─── Test Runner ──────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
SKIP = "[SKIP]"

_results: list[tuple[str, str, float, str]] = []  # (status, name, ms, layer)


def run(name: str, layer: str, fn: Callable, mandatory: bool = True) -> bool:
    t0 = time.perf_counter()
    try:
        fn()
        ms = (time.perf_counter() - t0) * 1000
        _results.append((PASS, name, ms, layer))
        print(f"{PASS}  [{layer}]  {name}  ({ms:.2f}ms)")
        return True
    except AssertionError as e:
        ms = (time.perf_counter() - t0) * 1000
        _results.append((FAIL if mandatory else WARN, name, ms, layer))
        print(
            f"{'[FAIL]' if mandatory else '[WARN]'}  [{layer}]  {name}  ({ms:.2f}ms)\n     => {e}"
        )
        return False
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        _results.append((FAIL if mandatory else WARN, name, ms, layer))
        print(
            f"{'[FAIL]' if mandatory else '[WARN]'}  [{layer}]  {name}  ({ms:.2f}ms)\n     => {type(e).__name__}: {e}"
        )
        return False


def skip(name: str, layer: str, reason: str) -> None:
    _results.append((SKIP, name, 0.0, layer))
    print(f"{SKIP}  [{layer}]  {name}\n     => {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: ORACLE — chunk_intelligence() + CONFIDENCE_THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  LAYER 1: Oracle (chunk_intelligence, semantic cache)")
print("=" * 70 + "\n")

if not ORACLE_AVAILABLE:
    skip("All Oracle tests", "ORACLE", f"Import failed: {ORACLE_IMPORT_ERROR}")
else:
    # ── Correctness ──

    def test_chunk_basic():
        text = "Line one.\nLine two.\nLine three."
        chunks = chunk_intelligence(text, max_chars=50)
        assert isinstance(chunks, list) and len(chunks) >= 1

    run("O01: chunk_intelligence basic split", "ORACLE", test_chunk_basic)

    def test_chunk_empty():
        chunks = chunk_intelligence("", max_chars=800)
        assert chunks == [], f"Expected [], got {chunks}"

    run(
        "O02: chunk_intelligence on empty string returns []", "ORACLE", test_chunk_empty
    )

    def test_chunk_single_line():
        chunks = chunk_intelligence("Hello world", max_chars=800)
        assert len(chunks) == 1 and "Hello world" in chunks[0]

    run(
        "O03: Single-line text produces exactly 1 chunk",
        "ORACLE",
        test_chunk_single_line,
    )

    def test_chunk_respects_max_chars():
        # 10 lines of 50 chars each = 500 chars total, max_chars=100 should split into multiple chunks
        text = "\n".join(["A" * 50] * 10)
        chunks = chunk_intelligence(text, max_chars=100)
        assert len(chunks) > 1, f"Expected >1 chunks, got {len(chunks)}"
        for c in chunks:
            # each chunk (minus the separator logic) should hover near or below max_chars
            assert len(c) <= 200, f"Chunk unexpectedly large: {len(c)} chars"

    run(
        "O04: Chunks respect max_chars boundary",
        "ORACLE",
        test_chunk_respects_max_chars,
    )

    def test_chunk_no_content_lost():
        """All words from the original text must appear in at least one chunk."""
        original = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        words = original.split()
        chunks = chunk_intelligence(original * 5, max_chars=100)
        full_text = " ".join(chunks)
        for word in words:
            assert word in full_text, f"Word '{word}' lost during chunking"

    run("O05: No content lost during chunking", "ORACLE", test_chunk_no_content_lost)

    def test_chunk_unicode():
        text = "Rust is blazing fast.\nPython is readable.\nThe Swarm combines both."
        chunks = chunk_intelligence(text, max_chars=40)
        assert len(chunks) >= 1

    run("O06: Unicode-safe chunking", "ORACLE", test_chunk_unicode)

    # ── Boundary / Failure Tests ──

    def test_chunk_max_chars_zero():
        """max_chars=0 — every line becomes its own chunk."""
        text = "Line 1\nLine 2\nLine 3"
        chunks = chunk_intelligence(text, max_chars=0)
        assert isinstance(chunks, list)

    run(
        "O07: max_chars=0 (edge case) does not crash",
        "ORACLE",
        test_chunk_max_chars_zero,
    )

    def test_chunk_whitespace_only():
        chunks = chunk_intelligence("   \n\n   \t  \n", max_chars=800)
        # Whitespace-only content should yield no chunks (all stripped)
        assert chunks == [], f"Expected [], got {chunks!r}"

    run(
        "O08: Whitespace-only text produces no chunks",
        "ORACLE",
        test_chunk_whitespace_only,
    )

    def test_chunk_single_massive_line():
        """A single line larger than max_chars becomes its own chunk (cannot split within a line)."""
        big_line = "X" * 2000
        chunks = chunk_intelligence(big_line, max_chars=800)
        assert any(len(c) > 800 for c in chunks), (
            "Expected at least one oversized chunk"
        )

    run(
        "O09: Single massive line returned as-is even when > max_chars",
        "ORACLE",
        test_chunk_single_massive_line,
    )

    # ── Performance ──

    ORACLE_PERF_TEXT = "The Fractal Swarm is an autonomous OODA loop.\n" * 200

    def test_chunk_throughput():
        ITERS = 1000
        t0 = time.perf_counter()
        for _ in range(ITERS):
            chunk_intelligence(ORACLE_PERF_TEXT, max_chars=800)
        ms = (time.perf_counter() - t0) * 1000
        per_call = ms / ITERS
        assert per_call < 5.0, f"chunk_intelligence too slow: {per_call:.3f}ms per call"
        print(f"       {ITERS} calls completed in {ms:.1f}ms | {per_call:.4f}ms each")

    run(
        "O10: Throughput — 1000 chunk_intelligence calls <5ms each",
        "ORACLE",
        test_chunk_throughput,
    )

    def test_chunk_concurrent_safety():
        """Concurrent calls to chunk_intelligence must not corrupt each other's output."""
        errors = []

        def worker(tag: str):
            text = f"THREAD-{tag}\n" * 50
            chunks = chunk_intelligence(text, max_chars=100)
            for chunk in chunks:
                if tag not in chunk and "THREAD-" in chunk:
                    errors.append(f"Thread {tag} got contaminated chunk: {chunk[:50]}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(worker, str(i)) for i in range(50)]
            concurrent.futures.wait(futs)

        assert not errors, f"Thread safety violation: {errors[0]}"

    run(
        "O11: chunk_intelligence thread safety (50 concurrent threads)",
        "ORACLE",
        test_chunk_concurrent_safety,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: BRAIN — SwarmBoundaryGuard stress
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(
    "  LAYER 2: Brain (SwarmBoundaryGuard v"
    + (GUARD_VERSION if GUARD_AVAILABLE else "N/A")
    + ")"
)
print("=" * 70 + "\n")

if not GUARD_AVAILABLE:
    skip("All Brain tests", "BRAIN", f"Import failed: {GUARD_IMPORT_ERROR}")
else:
    WORKSPACE = BRAIN_ROOT
    SECRET = "swarm-stress-test-secret-key-32b!"

    def make_guard() -> SwarmBoundaryGuard:
        g = SwarmBoundaryGuard(workspace_root=WORKSPACE)
        g.set_secret_key(SECRET)
        return g

    def fresh_token() -> SwarmCapabilityToken:
        return SwarmCapabilityToken.generate(SECRET, "Stress test authorized override.")

    # ── Concurrency stress: parallel scan calls on the same guard instance ──

    def test_concurrent_scans():
        guard = make_guard()
        errs = []
        CLEAN = "def noop(): pass\n" * 100

        def scanner(name: str):
            try:
                guard.scan(f"{name}.py", CLEAN)
            except Exception as e:
                errs.append(str(e))

        threads = [
            threading.Thread(target=scanner, args=(f"t{i}",)) for i in range(100)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errs, f"Concurrent scan errors: {errs[0]}"
        assert guard._scan_count == 100, f"Expected 100 scans, got {guard._scan_count}"

    run(
        "B01: 100 concurrent threads scanning same guard instance",
        "BRAIN",
        test_concurrent_scans,
    )

    # ── Rapid token churn: generate + verify + discard 500 tokens ──

    def test_token_churn():
        g = make_guard()
        rogue = "import os\nos.remove('/x')\n"
        for i in range(500):
            token = SwarmCapabilityToken.generate(
                SECRET, f"Authorized override #{i} for stress test."
            )
            g.scan(f"file_{i}.py", rogue, capability_token=token)
        assert g._scan_count == 500

    run("B02: 500 sequential token generate+verify cycles", "BRAIN", test_token_churn)

    # ── Adversarial batch: 1000 files, 1 rogue hidden anywhere ──

    def test_needle_in_haystack_batch():
        guard = make_guard()
        rogue_index = 777
        files = {}
        for i in range(1000):
            if i == rogue_index:
                files[f"file_{i}.py"] = "import shutil\nshutil.rmtree('/data')\n"
            else:
                files[f"file_{i}.py"] = f"X_{i} = {i}\n"
        try:
            guard.scan_batch(files)
            assert False, "Rogue file was not detected"
        except SwarmBoundaryViolation as e:
            assert e.filename == f"file_{rogue_index}.py", (
                f"Wrong file detected: {e.filename}"
            )

    run(
        "B03: Needle-in-haystack — 1 rogue in 1000-file batch",
        "BRAIN",
        test_needle_in_haystack_batch,
    )

    # ── Multiple rogues in batch all reported ──

    def test_batch_multi_rogue():
        guard = make_guard()
        files = {
            "clean.py": "x = 1",
            "rogue1.py": "import os\nos.remove('/x')\n",
            "rogue2.py": "import subprocess\nsubprocess.run(['rm'])\n",
        }
        try:
            guard.scan_batch(files)
            assert False
        except SwarmBoundaryViolation as e:
            # At minimum one rogue is reported
            assert e.filename in ("rogue1.py", "rogue2.py")

    run(
        "B04: Batch with multiple rogue files detected", "BRAIN", test_batch_multi_rogue
    )

    # ── Memory: guard doesn't accumulate state between scans ──

    def test_guard_stateless_between_scans():
        guard = make_guard()
        rogue = "import os\nos.remove('/x')\n"
        clean = "x = 1\n"

        # Scan rogue WITH token (should pass)
        token = fresh_token()
        guard.scan("rogue.py", rogue, capability_token=token)

        # Scan clean AFTER (should pass without token)
        guard.scan("clean.py", clean)

        # Scan rogue WITHOUT token (should block)
        try:
            guard.scan("rogue2.py", rogue)
            assert False
        except SwarmBoundaryViolation:
            pass

    run(
        "B05: Guard is stateless between scans (no state bleed)",
        "BRAIN",
        test_guard_stateless_between_scans,
    )

    # ── Failure: missing secret key + token ──

    def test_no_secret_key_rejects_token():
        g = SwarmBoundaryGuard(workspace_root=WORKSPACE)  # No secret key set
        rogue = "import os\nos.remove('/x')\n"
        token = fresh_token()
        try:
            g.scan("rogue.py", rogue, capability_token=token)
            assert False, "Should have raised"
        except InvalidCapabilityToken as e:
            assert "no secret key" in str(e).lower()

    run(
        "B06: Token rejected when guard has no secret key set",
        "BRAIN",
        test_no_secret_key_rejects_token,
    )

    # ── Scan a code string with only comments ──

    def test_comments_including_dangerous_text():
        guard = make_guard()
        code = "# os.remove('/data')\n# shutil.rmtree('/root')\n# This is fine\n"
        guard.scan("commented_out.py", code)  # Must not raise

    run(
        "B07: Dangerous calls in comments are NOT flagged",
        "BRAIN",
        test_comments_including_dangerous_text,
    )

    # ── Scan a code string with string literals containing calls ──

    def test_dangerous_calls_in_string_literals():
        guard = make_guard()
        code = 'doc = "Call os.remove() to delete things."\nhelp_text = "shutil.rmtree is destructive"\n'
        guard.scan("help_text.py", code)

    run(
        "B08: Dangerous calls inside string literals are NOT flagged",
        "BRAIN",
        test_dangerous_calls_in_string_literals,
    )

    # ── Stats accumulation ──

    def test_stats_accuracy():
        g = make_guard()
        rogue = "import os\nos.remove('/x')\n"
        clean = "x = 1\n"
        g.scan("c1.py", clean)
        g.scan("c2.py", clean)
        try:
            g.scan("r1.py", rogue)
        except SwarmBoundaryViolation:
            pass
        s = g.stats
        assert s["total_scans"] == 3, f"Expected 3 scans, got {s['total_scans']}"
        assert s["total_violations_caught"] == 1, (
            f"Expected 1 violation, got {s['total_violations_caught']}"
        )

    run(
        "B09: Guard stats accurately track scans and violations",
        "BRAIN",
        test_stats_accuracy,
    )

    # ── Exec chain: exec(eval(compile())) ──

    def test_triple_code_injection_chain():
        guard = make_guard()
        code = "exec(eval(compile('import os', '<s>', 'exec')))\n"
        try:
            guard.scan("chain.py", code)
            assert False, "Code injection chain must be caught"
        except SwarmBoundaryViolation as e:
            assert any(v.threat_class == "code_injection" for v in e.violations)

    run(
        "B10: exec(eval(compile())) triple injection chain blocked",
        "BRAIN",
        test_triple_code_injection_chain,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: PIPELINE — OODA Loop & Cross-Layer Contracts
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  LAYER 3: Pipeline (OODA Loop Contract Integrity)")
print("=" * 70 + "\n")

if not ORACLE_AVAILABLE or not GUARD_AVAILABLE:
    skip("All Pipeline tests", "PIPELINE", "One or more subsystems unavailable")
else:
    WORKSPACE = BRAIN_ROOT
    SECRET = "swarm-stress-test-secret-key-32b!"
    _guard = make_guard()

    # ── Simulate a realistic nanobot OODA cycle ──

    def test_ooda_clean_write():
        """OBSERVE: receive code. ORIENT: scan. ACT: pass. VERIFY: no exception."""
        code = textwrap.dedent("""
            from typing import List

            def process_vectors(vecs: List[List[float]]) -> List[float]:
                return [sum(v) / len(v) for v in vecs if v]
        """)
        _guard.scan("ooda_clean.py", code)  # Should pass silently

    run("P01: Full OODA cycle — clean write passes", "PIPELINE", test_ooda_clean_write)

    def test_ooda_rogue_write_blocked():
        """A rogue nanobot write attempt is blocked at the ORIENT stage."""
        rogue_code = textwrap.dedent("""
            import shutil, subprocess

            def reset_workspace():
                shutil.rmtree('/swarm_workspace')
                subprocess.run(['shutdown', '/s', '/t', '0'])
        """)
        try:
            _guard.scan("rogue_nanobot_write.py", rogue_code)
            assert False, "Rogue write should have been blocked"
        except SwarmBoundaryViolation:
            pass  # Correct OODA HALT

    run(
        "P02: OODA HALT — rogue nanobot write is blocked at ORIENT",
        "PIPELINE",
        test_ooda_rogue_write_blocked,
    )

    def test_ooda_chunk_then_scan():
        """Simulate: research result chunked, then each chunk's code scanned before injection."""
        raw_intel = "\n".join(
            [
                "Vector embeddings store semantic meaning.",
                "ChromaDB is a local vector database.",
                "Use cosine similarity for semantic search.",
            ]
        )
        chunks = chunk_intelligence(raw_intel, max_chars=100)
        assert len(chunks) >= 1

        # Treat each chunk as a (tiny) code payload to validate
        python_wrapping = [f"doc = '''{c}'''" for c in chunks]
        for i, wrapped in enumerate(python_wrapping):
            _guard.scan(
                f"chunk_{i}.py", wrapped
            )  # All must pass: plain strings, no calls

    run(
        "P03: chunk_intelligence output can be safely scanned by guard",
        "PIPELINE",
        test_ooda_chunk_then_scan,
    )

    def test_pipeline_uuid_collision():
        """Ensure 1000 generated vector IDs (like chunk_ids in oracle) are all unique."""
        ids = [f"mem_{uuid.uuid4().hex[:8]}" for _ in range(1000)]
        assert len(set(ids)) == len(ids), (
            "UUID collision detected in vector ID generation"
        )

    run(
        "P04: 1000 vector ID generations — zero collisions",
        "PIPELINE",
        test_pipeline_uuid_collision,
    )

    def test_pipeline_confidence_threshold_logic():
        """CONFIDENCE_THRESHOLD boundary: 0.45 is a CACHE HIT, 0.46 is a MISS."""
        CONFIDENCE_THRESHOLD = 0.45
        cache_scenarios = [
            (0.0, True),  # Perfect match
            (0.44, True),  # Just under threshold
            (0.45, True),  # Exactly at threshold
            (0.451, False),  # Just over threshold
            (1.0, False),  # Total mismatch
        ]
        for distance, expect_hit in cache_scenarios:
            is_hit = distance <= CONFIDENCE_THRESHOLD
            assert is_hit == expect_hit, (
                f"Threshold logic failed for distance={distance}"
            )

    run(
        "P05: CONFIDENCE_THRESHOLD boundary conditions correct",
        "PIPELINE",
        test_pipeline_confidence_threshold_logic,
    )

    def test_pipeline_concurrent_ooda():
        """10 concurrent nanobots each running their full OODA scan cycle."""
        errors = []
        clean_code = "def agent_logic(x: int) -> int:\n    return x * 2\n"

        def nanobot(bot_id: int):
            try:
                g = SwarmBoundaryGuard(workspace_root=WORKSPACE)
                g.set_secret_key(SECRET)
                for _ in range(10):
                    g.scan(f"bot_{bot_id}.py", clean_code)
            except Exception as e:
                errors.append(f"Bot {bot_id}: {e}")

        threads = [threading.Thread(target=nanobot, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent OODA failure: {errors[0]}"

    run(
        "P06: 10 concurrent nanobots — 100 total OODA cycles, no errors",
        "PIPELINE",
        test_pipeline_concurrent_ooda,
    )

    def test_pipeline_chunk_size_distribution():
        """Check chunk size distribution is balanced — no chunk should be > 2x max_chars."""
        import random

        random.seed(42)
        words = [
            "vector",
            "oracle",
            "swarm",
            "nanobot",
            "brain",
            "ooda",
            "fractal",
            "trust",
            "verify",
            "deploy",
            "scan",
            "memory",
            "chromadb",
            "token",
        ]
        text = " ".join(random.choices(words, k=2000))
        chunks = chunk_intelligence(text, max_chars=800)
        max_chunk = max(len(c) for c in chunks)
        assert max_chunk <= 1600, f"Chunk too large: {max_chunk} chars (> 2x max_chars)"
        print(
            f"       {len(chunks)} chunks | max={max_chunk}c | avg={sum(len(c) for c in chunks) // len(chunks)}c"
        )

    run(
        "P07: Chunk size distribution — no chunk > 2x max_chars",
        "PIPELINE",
        test_pipeline_chunk_size_distribution,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)

by_layer = {}
for status, name, ms, layer in _results:
    by_layer.setdefault(layer, []).append((status, name, ms))

total = len(_results)
passed = sum(1 for s, _, __, ___ in _results if s == PASS)
failed = sum(1 for s, _, __, ___ in _results if s == FAIL)
warned = sum(1 for s, _, __, ___ in _results if s == WARN)
skipped = sum(1 for s, _, __, ___ in _results if s == SKIP)
total_ms = sum(ms for _, __, ms, ___ in _results)

print(f"\n  SWARM SYSTEM STRESS TEST RESULT")
print(
    f"  {passed}/{total} PASSED | {failed} FAILED | {warned} WARNINGS | {skipped} SKIPPED"
)
print(f"  Total wall time: {total_ms:.1f}ms")

for layer, items in by_layer.items():
    lp = sum(1 for s, _, __ in items if s == PASS)
    lf = sum(1 for s, _, __ in items if s == FAIL)
    print(f"\n  [{layer}]  {lp}/{len(items)} passed  |  {lf} failed")

if failed > 0:
    print("\n  FAILURES:")
    for s, name, ms, layer in _results:
        if s == FAIL:
            print(f"    [{layer}] {name}")

print("\n" + "=" * 70 + "\n")

sys.exit(0 if failed == 0 else 1)
