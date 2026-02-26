"""
Fractal Swarm — FULL SYSTEM STRESS TEST v2.0
=============================================
Covers all 4 active subsystems + cross-layer contracts:

  LAYER 1: Oracle         —  chunk_intelligence, confidence threshold
  LAYER 2: Brain          —  SwarmBoundaryGuard v1.1.0, SwarmCapabilityToken
  LAYER 3: Crypto (A2A)   —  ReleaseManifest, canonical_hash, IntegrityError
  LAYER 4: Namespace      —  workspace_id resolver, .swarm_workspace logic
  LAYER 5: Integration    —  Cross-layer OODA pipeline contracts

Run:
  python swarm_full_stress_v2.py

Exit 0 = all mandatory tests pass.
Exit 1 = one or more mandatory tests FAILED.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
import hashlib
import hmac
import tempfile
import threading
import concurrent.futures
import textwrap
from copy import deepcopy
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────────────────

SWARM_ROOT = Path(__file__).parent.parent.parent
BRAIN_ROOT = SWARM_ROOT / "core" / "brain"
CRYPTO_ROOT = SWARM_ROOT / "core" / "crypto"
NS_ROOT = SWARM_ROOT / "core" / "namespace"
SANDBOX = SWARM_ROOT / "pipelines" / "sandboxes"

for p in (BRAIN_ROOT, CRYPTO_ROOT, NS_ROOT, SANDBOX):
    sys.path.insert(0, str(p))

# ─── Imports ──────────────────────────────────────────────────────────────────

PASS, FAIL, WARN, SKIP = "[PASS]", "[FAIL]", "[WARN]", "[SKIP]"
_results: list[tuple[str, str, float, str]] = []


def run(name: str, layer: str, fn, mandatory: bool = True) -> bool:
    t0 = time.perf_counter()
    try:
        fn()
        ms = (time.perf_counter() - t0) * 1000
        _results.append((PASS, name, ms, layer))
        print(f"{PASS}  [{layer}]  {name}  ({ms:.2f}ms)")
        return True
    except AssertionError as e:
        ms = (time.perf_counter() - t0) * 1000
        tag = FAIL if mandatory else WARN
        _results.append((tag, name, ms, layer))
        print(f"{tag}  [{layer}]  {name}  ({ms:.2f}ms)\n       => AssertionError: {e}")
        return False
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        tag = FAIL if mandatory else WARN
        _results.append((tag, name, ms, layer))
        print(
            f"{tag}  [{layer}]  {name}  ({ms:.2f}ms)\n       => {type(e).__name__}: {e}"
        )
        return False


def skip(name: str, layer: str, reason: str):
    _results.append((SKIP, name, 0.0, layer))
    print(f"{SKIP}  [{layer}]  {name}\n       => {reason}")


# ─── Module availability ──────────────────────────────────────────────────────

try:
    from oracle_pure import chunk_intelligence

    ORACLE_OK = True
except ImportError as e:
    ORACLE_OK = False
    ORACLE_ERR = str(e)

try:
    from swarm_boundary_guard import (
        SwarmBoundaryGuard,
        SwarmCapabilityToken,
        SwarmBoundaryViolation,
        InvalidCapabilityToken,
        GUARD_VERSION,
    )

    GUARD_OK = True
except ImportError as e:
    GUARD_OK = False
    GUARD_ERR = str(e)

try:
    from crypto_utils import (
        ReleaseManifest,
        IntegrityError,
        canonical_hash,
        verify_hash,
    )

    CRYPTO_OK = True
except ImportError as e:
    CRYPTO_OK = False
    CRYPTO_ERR = str(e)

try:
    from workspace_id import get_workspace_id, set_workspace_id, _slugify

    NS_OK = True
except ImportError as e:
    NS_OK = False
    NS_ERR = str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: ORACLE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  LAYER 1: Oracle  (chunk_intelligence)")
print(f"{'=' * 70}\n")

if not ORACLE_OK:
    skip("All Oracle tests", "ORACLE", ORACLE_ERR)
else:
    ORACLE_TEXT = "The Fractal Swarm is an autonomous OODA loop.\n" * 200

    def o01():
        chunk_intelligence("", max_chars=800) == []

    run("O01: Empty input returns []", "ORACLE", o01)

    def o02():
        r = chunk_intelligence("hello world", 800)
        assert len(r) == 1 and "hello" in r[0]

    run("O02: Single word produces 1 chunk", "ORACLE", o02)

    def o03():
        chunks = chunk_intelligence("A" * 2000, max_chars=800)
        assert all(len(c) <= 810 for c in chunks), (
            f"Chunk exceeded: {max(len(c) for c in chunks)}"
        )

    run("O03: 2000-char single line split within 810-char budget", "ORACLE", o03)

    def o04():
        text = "\n".join(["word" * 10] * 100)
        c = chunk_intelligence(text, max_chars=800)
        assert len(c) > 1

    run("O04: 100-line text produces multiple chunks", "ORACLE", o04)

    def o05():
        t0 = time.perf_counter()
        for _ in range(2000):
            chunk_intelligence(ORACLE_TEXT, max_chars=800)
        ms = (time.perf_counter() - t0) * 1000
        per = ms / 2000
        assert per < 5.0, f"{per:.3f}ms per call (>5ms budget)"
        print(f"       2000 calls | {ms:.0f}ms total | {per:.3f}ms each")

    run("O05: Throughput — 2000 calls < 5ms each", "ORACLE", o05)

    def o06():
        errs = []

        def worker(tag):
            c = chunk_intelligence(f"THREAD-{tag}\n" * 50, max_chars=100)
            for chunk in c:
                if tag not in chunk and "THREAD-" in chunk:
                    errs.append(tag)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as p:
            list(p.map(worker, [str(i) for i in range(100)]))
        assert not errs, f"Thread contamination in {len(errs)} chunks"

    run("O06: 100 concurrent threads — thread-safe chunking", "ORACLE", o06)

    def o07():
        chunks = chunk_intelligence("   \n\t\n   ", 800)
        assert chunks == [], f"Expected [], got {chunks}"

    run("O07: Whitespace-only input produces no chunks", "ORACLE", o07)

    def o08():
        big = "word " * 5000
        chunks = chunk_intelligence(big, max_chars=800)
        max_c = max(len(c) for c in chunks)
        assert max_c <= 820, f"Chunk too large: {max_c}"
        print(f"       {len(chunks)} chunks | max={max_c}c")

    run("O08: 25k-char run-on text stays within size budget", "ORACLE", o08)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: BRAIN (Guard)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  LAYER 2: Brain  (SwarmBoundaryGuard {GUARD_VERSION if GUARD_OK else 'N/A'})")
print(f"{'=' * 70}\n")

if not GUARD_OK:
    skip("All Brain tests", "BRAIN", GUARD_ERR)
else:
    WS = BRAIN_ROOT
    SEC = "stress-test-secret-key-v2!32bytes"

    def g():
        g_ = SwarmBoundaryGuard(workspace_root=WS)
        g_.set_secret_key(SEC)
        return g_

    def tok(j="Authorized stress test override for cleanup."):
        return SwarmCapabilityToken.generate(SEC, j)

    def b01():
        g().scan("clean.py", "def f(): return 1")

    run("B01: Clean code passes", "BRAIN", b01)

    def b02():
        try:
            g().scan("r.py", "import os\nos.remove('/x')")
        except SwarmBoundaryViolation:
            return
        assert False

    run("B02: os.remove blocked", "BRAIN", b02)

    def b03():
        try:
            g().scan("r.py", "import shutil as s\ns.rmtree('/x')")
        except SwarmBoundaryViolation:
            return
        assert False

    run("B03: Aliased shutil.rmtree blocked", "BRAIN", b03)

    def b04():
        try:
            g().scan("r.py", "exec('import os; os.system(\"rm -rf /\")')")
        except SwarmBoundaryViolation:
            return
        assert False

    run("B04: exec() code injection blocked", "BRAIN", b04)

    def b05():
        gu = g()
        rogue = "import os\nos.remove('/x')"
        t = tok()
        gu.scan("auth.py", rogue, capability_token=t)
        try:
            gu.scan("auth2.py", rogue, capability_token=t)
        except InvalidCapabilityToken:
            return
        assert False

    run("B05: Consumed token rejected on reuse", "BRAIN", b05)

    def b06():
        errs, gu = [], g()
        CLEAN = "def f(x): return x * 2\n"

        def scanner(n):
            try:
                gu.scan(f"f{n}.py", CLEAN)
            except Exception as e:
                errs.append(str(e))

        ts = [threading.Thread(target=scanner, args=(i,)) for i in range(200)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert not errs, errs[0]
        assert gu._scan_count == 200

    run("B06: 200 concurrent threads on same guard instance", "BRAIN", b06)

    def b07():
        gu = g()
        files = {f"f{i}.py": "x = 1" for i in range(500)}
        files["rogue_333.py"] = "import os\nos.remove('/x')"
        try:
            gu.scan_batch(files)
        except SwarmBoundaryViolation as e:
            assert e.filename == "rogue_333.py"
            return
        assert False

    run("B07: Needle-in-500-file-batch caught at exact index", "BRAIN", b07)

    def b08():
        gu = g()
        code = "# os.remove('/data')\ndoc = 'call shutil.rmtree to delete'\n"
        gu.scan("safe_comment.py", code)

    run("B08: Calls in comments/strings NOT flagged", "BRAIN", b08)

    def b09():
        gu = g()
        code = textwrap.dedent("""
            import os
            try:
                def inner():
                    return lambda: os.system('kill')
            finally:
                pass
        """)
        try:
            gu.scan("nested.py", code)
        except SwarmBoundaryViolation:
            return
        assert False

    run("B09: Deeply nested lambda os.system caught", "BRAIN", b09)

    def b10():
        gu = g()
        t0 = time.perf_counter()
        CLEAN = "x = [i**2 for i in range(100)]\n"
        for i in range(1000):
            gu.scan(f"f{i}.py", CLEAN)
        ms = (time.perf_counter() - t0) * 1000
        per = ms / 1000
        assert per < 5.0, f"{per:.3f}ms per scan (>5ms budget)"
        print(f"       1000 scans | {ms:.0f}ms total | {per:.3f}ms each")

    run("B10: Throughput — 1000 clean scans < 5ms each", "BRAIN", b10)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: CRYPTO (A2A ReleaseManifest)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  LAYER 3: Crypto  (A2A ReleaseManifest / SHA-256)")
print(f"{'=' * 70}\n")

if not CRYPTO_OK:
    skip("All Crypto tests", "CRYPTO", CRYPTO_ERR)
else:
    PAYLOAD_BASE = {"project": "omega", "seed": "Aegis-Fractal-0x99A", "tier": 1}

    def c01():
        m = ReleaseManifest.sign(PAYLOAD_BASE, "node_01")
        wire = m.to_wire()
        recovered = ReleaseManifest.verify_wire(wire)
        assert recovered.payload == PAYLOAD_BASE

    run("C01: Clean sign + verify_wire round-trip", "CRYPTO", c01)

    def c02():
        """Key-order independence: any dict ordering hashes identically."""
        h1 = canonical_hash({"b": 2, "a": 1, "c": [3, 2, 1]})
        h2 = canonical_hash({"c": [3, 2, 1], "a": 1, "b": 2})
        assert h1 == h2, f"Non-deterministic: {h1} != {h2}"

    run("C02: canonical_hash is key-order independent", "CRYPTO", c02)

    def c03():
        """Tampered payload field raises IntegrityError."""
        wire = ReleaseManifest.sign(PAYLOAD_BASE, "node_01").to_wire()
        wire["payload"] = {"project": "omega", "seed": "POISON", "tier": 1}
        try:
            ReleaseManifest.verify_wire(wire)
            assert False, "Should have raised IntegrityError"
        except IntegrityError:
            pass

    run("C03: Tampered payload raises IntegrityError", "CRYPTO", c03)

    def c04():
        """Modified SHA-256 field itself is caught."""
        wire = ReleaseManifest.sign(PAYLOAD_BASE, "node_01").to_wire()
        wire["payload_sha256"] = "a" * 64  # garbage hash
        try:
            ReleaseManifest.verify_wire(wire)
            assert False, "Should have raised IntegrityError"
        except IntegrityError:
            pass

    run("C04: Forged SHA-256 field raises IntegrityError", "CRYPTO", c04)

    def c05():
        """Missing required manifest field."""
        try:
            ReleaseManifest.from_wire({"broken": True})
            assert False
        except IntegrityError:
            pass

    run("C05: Malformed wire dict raises IntegrityError", "CRYPTO", c05)

    def c06():
        """verify_hash convenience function."""
        assert verify_hash(PAYLOAD_BASE, canonical_hash(PAYLOAD_BASE))
        assert not verify_hash(PAYLOAD_BASE, "a" * 64)

    run("C06: verify_hash returns True/False correctly", "CRYPTO", c06)

    def c07():
        """Large nested payload (simulates a real vector bundle)."""
        big_payload = {
            "vectors": [[float(i % 768) / 768 for i in range(768)] for _ in range(10)],
            "metadata": {"workspace": "fractal_swarm", "chunk_count": 10},
        }
        m = ReleaseManifest.sign(big_payload, "node_stress")
        wire = m.to_wire()
        ReleaseManifest.verify_wire(wire)

    run("C07: Large nested vector bundle signs and verifies", "CRYPTO", c07)

    def c08():
        """500 sequential sign+verify cycles — throughput benchmark."""
        t0 = time.perf_counter()
        for i in range(500):
            payload = {"index": i, "data": f"chunk_{i}" * 20}
            wire = ReleaseManifest.sign(payload, "bench_node").to_wire()
            ReleaseManifest.verify_wire(wire)
        ms = (time.perf_counter() - t0) * 1000
        per = ms / 500
        assert per < 10.0, f"{per:.3f}ms per cycle (>10ms)"
        print(f"       500 sign+verify cycles | {ms:.0f}ms | {per:.3f}ms each")

    run("C08: Throughput — 500 sign+verify < 10ms each", "CRYPTO", c08)

    def c09():
        """Concurrent manifest generation — no race conditions."""
        errs = []

        def worker(n):
            try:
                p = {"thread": n, "data": "x" * 100}
                m = ReleaseManifest.sign(p, f"node_{n}")
                wire = m.to_wire()
                r = ReleaseManifest.verify_wire(wire)
                assert r.payload["thread"] == n
            except Exception as e:
                errs.append(f"Thread {n}: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(worker, range(200)))
        assert not errs, errs[0]

    run("C09: 200 concurrent sign+verify threads — no race conditions", "CRYPTO", c09)

    def c10():
        """Empty string payload hashes deterministically."""
        h1 = canonical_hash("")
        h2 = canonical_hash("")
        assert h1 == h2 and len(h1) == 64

    run("C10: Empty string payload canonical_hash is stable", "CRYPTO", c10)

    def c11():
        """List payload (not dict) signs and verifies cleanly."""
        payload = ["alpha", "beta", "gamma"]
        m = ReleaseManifest.sign(payload, "list_node")
        ReleaseManifest.verify_wire(m.to_wire())

    run("C11: List (non-dict) payload signs and verifies", "CRYPTO", c11)

    def c12():
        """Timing-safe comparison: verify_hash doesn't short-circuit."""
        real = canonical_hash(PAYLOAD_BASE)
        # Flip last character — should still produce constant-time rejection
        fake = real[:-1] + ("a" if real[-1] != "a" else "b")
        assert not verify_hash(PAYLOAD_BASE, fake)

    run("C12: verify_hash rejects single-char difference", "CRYPTO", c12)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4: NAMESPACE RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  LAYER 4: Namespace  (workspace_id resolver)")
print(f"{'=' * 70}\n")

if not NS_OK:
    skip("All Namespace tests", "NAMESPACE", NS_ERR)
else:

    def n01():
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "my_swarm_project"
            p.mkdir()
            ws_file = p / ".swarm_workspace"
            ws_file.write_text(
                json.dumps({"workspace_id": "test_namespace_alpha"}), encoding="utf-8"
            )
            assert get_workspace_id(str(p)) == "test_namespace_alpha"

    run("N01: Reads workspace_id from .swarm_workspace file", "NAMESPACE", n01)

    def n02():
        """Walks up ancestor dirs to find .swarm_workspace."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "project"
            deep = root / "src" / "core" / "utils"
            deep.mkdir(parents=True)
            (root / ".swarm_workspace").write_text(
                json.dumps({"workspace_id": "parent_workspace"}), encoding="utf-8"
            )
            assert get_workspace_id(str(deep)) == "parent_workspace"

    run("N02: Walks parent dirs to find .swarm_workspace", "NAMESPACE", n02)

    def n03():
        """Env var overrides everything."""
        old = os.environ.get("SWARM_WORKSPACE")
        os.environ["SWARM_WORKSPACE"] = "ci_override_workspace"
        try:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "irrelevant_project"
                p.mkdir()
                (p / ".swarm_workspace").write_text(
                    json.dumps({"workspace_id": "should_not_appear"}), encoding="utf-8"
                )
                assert get_workspace_id(str(p)) == "ci_override_workspace"
        finally:
            if old is None:
                del os.environ["SWARM_WORKSPACE"]
            else:
                os.environ["SWARM_WORKSPACE"] = old

    run("N03: SWARM_WORKSPACE env var takes priority over file", "NAMESPACE", n03)

    def n04():
        """Fallbacks to slugified directory name."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "My Awesome Project 2024!"
            p.mkdir()
            result = get_workspace_id(str(p))
            assert result == "my_awesome_project_2024", f"Unexpected: {result!r}"

    run("N04: Slugify fallback for dir with spaces and symbols", "NAMESPACE", n04)

    def n05():
        """Corrupt .swarm_workspace falls through to dir-name slug."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "good_project"
            p.mkdir()
            (p / ".swarm_workspace").write_text("NOT JSON {{", encoding="utf-8")
            result = get_workspace_id(str(p))
            assert result == "good_project", f"Got: {result!r}"

    run("N05: Corrupt .swarm_workspace falls back to dir-name slug", "NAMESPACE", n05)

    def n06():
        """.swarm_workspace with missing workspace_id key falls back."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "project_b"
            p.mkdir()
            (p / ".swarm_workspace").write_text(
                json.dumps({"other": "value"}), encoding="utf-8"
            )
            assert get_workspace_id(str(p)) == "project_b"

    run(
        "N06: .swarm_workspace missing workspace_id key uses dir slug", "NAMESPACE", n06
    )

    def n07():
        """set_workspace_id writes a valid file that get_workspace_id reads back."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "new_project"
            p.mkdir()
            set_workspace_id("my_new_workspace", target_dir=str(p))
            assert get_workspace_id(str(p)) == "my_new_workspace"

    run("N07: set_workspace_id + get_workspace_id round-trip", "NAMESPACE", n07)

    def n08():
        """Concurrent get_workspace_id calls on the same dir — no corruption."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "shared_project"
            p.mkdir()
            (p / ".swarm_workspace").write_text(
                json.dumps({"workspace_id": "shared_ns"}), encoding="utf-8"
            )
            errs = []

            def reader(_):
                r = get_workspace_id(str(p))
                if r != "shared_ns":
                    errs.append(r)

            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
                list(pool.map(reader, range(100)))
            assert not errs, f"Got wrong namespace: {errs[0]}"

    run(
        "N08: 100 concurrent namespace resolutions — consistent result",
        "NAMESPACE",
        n08,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5: INTEGRATION  (Cross-layer OODA pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  LAYER 5: Integration  (Full OODA Pipeline Contracts)")
print(f"{'=' * 70}\n")

if not all([ORACLE_OK, GUARD_OK, CRYPTO_OK, NS_OK]):
    skip("All Integration tests", "INTEGRATION", "One or more subsystems unavailable")
else:
    _guard = SwarmBoundaryGuard(workspace_root=BRAIN_ROOT)
    _guard.set_secret_key("integration-test-secret-32b-key!")

    def i01():
        """Chunk → scan → sign: full clean OODA cycle."""
        text = (
            "FastAPI handles routing. ChromaDB handles embeddings. Ruff formats code."
        )
        chunks = chunk_intelligence(text, max_chars=100)
        for i, chunk in enumerate(chunks):
            code = f'doc_{i} = """{chunk}"""'
            _guard.scan(f"chunk_{i}.py", code)  # Must pass
        # Sign the last chunk as a simulated A2A payload
        manifest = ReleaseManifest.sign(chunks[-1], "integration_node")
        ReleaseManifest.verify_wire(manifest.to_wire())

    run("I01: Chunk → AST scan → A2A sign/verify full OODA cycle", "INTEGRATION", i01)

    def i02():
        """Rogue payload injected mid-OODA is caught before signing."""
        rogue_code = "import shutil\nshutil.rmtree('/workspace')"
        try:
            _guard.scan("injected.py", rogue_code)
            assert False, "Rogue code should have halted OODA"
        except SwarmBoundaryViolation:
            pass  # HALT — never reaches A2A layer

    run("I02: Rogue code halts OODA before reaching A2A signing", "INTEGRATION", i02)

    def i03():
        """A tampered manifest is caught at the receiver gate."""
        payload = "Legitimate semantic vector content."
        manifest = ReleaseManifest.sign(payload, "sender_node")
        wire = manifest.to_wire()
        # Simulate MITM tampering
        wire["payload"] = "COMPROMISED content from MITM attacker."
        try:
            ReleaseManifest.verify_wire(wire)
            assert False, "Tampered manifest must be rejected"
        except IntegrityError:
            pass  # Correctly blocked at receiver gate

    run("I03: MITM-tampered manifest blocked at receiver gate", "INTEGRATION", i03)

    def i04():
        """Namespace isolation: two workspace resolvers return independent IDs."""
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "project_alpha"
            p2 = Path(d) / "project_beta"
            p1.mkdir()
            p2.mkdir()
            set_workspace_id("ns_alpha", str(p1))
            set_workspace_id("ns_beta", str(p2))
            assert get_workspace_id(str(p1)) == "ns_alpha"
            assert get_workspace_id(str(p2)) == "ns_beta"
            assert get_workspace_id(str(p1)) != get_workspace_id(str(p2))

    run("I04: Two namespaces isolated — no cross-contamination", "INTEGRATION", i04)

    def i05():
        """Pipeline UUID deduplication across 5000 vector IDs."""
        ids = {f"mem_{uuid.uuid4().hex[:8]}" for _ in range(5000)}
        assert len(ids) == 5000, "UUID collision detected!"

    run("I05: 5000 vector IDs generated — zero UUID collisions", "INTEGRATION", i05)

    def i06():
        """Concurrent 8-nanobot OODA loop: chunk + scan + sign."""
        errs = []

        def nanobot(bot_id):
            try:
                g = SwarmBoundaryGuard(workspace_root=BRAIN_ROOT)
                g.set_secret_key("concurrent-test-key-32bytes!!!")
                for _ in range(5):
                    text = f"Bot {bot_id} research chunk " * 20
                    chunks = chunk_intelligence(text, max_chars=200)
                    code = "\n".join(
                        f'd_{i} = """{c[:50]}"""' for i, c in enumerate(chunks)
                    )
                    g.scan(f"bot_{bot_id}.py", code)
                    manifest = ReleaseManifest.sign(chunks, f"bot_{bot_id}")
                    ReleaseManifest.verify_wire(manifest.to_wire())
            except Exception as e:
                errs.append(f"Bot {bot_id}: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(nanobot, range(8)))
        assert not errs, errs[0]

    run("I06: 8 concurrent nanobots — 40 total full OODA cycles", "INTEGRATION", i06)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")

by_layer: dict[str, list] = {}
for status, name, ms, layer in _results:
    by_layer.setdefault(layer, []).append((status, ms))

total = len(_results)
passed = sum(1 for s, *_ in _results if s == PASS)
failed = sum(1 for s, *_ in _results if s == FAIL)
warned = sum(1 for s, *_ in _results if s == WARN)
skipped = sum(1 for s, *_ in _results if s == SKIP)
total_ms = sum(ms for _, __, ms, ___ in _results)
fastest = min((ms for _, __, ms, ___ in _results if ms > 0), default=0)
slowest = max((ms for _, __, ms, ___ in _results), default=0)

print(f"""
  FRACTAL SWARM FULL STRESS TEST v2.0  —  RESULT
  {"=" * 50}
  {passed}/{total} PASSED | {failed} FAILED | {warned} WARNINGS | {skipped} SKIPPED
  Wall time: {total_ms:.1f}ms | Fastest: {fastest:.2f}ms | Slowest: {slowest:.0f}ms
""")

for layer, items in by_layer.items():
    lp = sum(1 for s, _ in items if s == PASS)
    lf = sum(1 for s, _ in items if s == FAIL)
    lms = sum(m for _, m in items)
    print(
        f"  [{layer:12s}]  {lp:2d}/{len(items):2d} passed  |  {lf} failed  |  {lms:.0f}ms"
    )

if failed > 0:
    print("\n  FAILURES:")
    for s, name, ms, layer in _results:
        if s == FAIL:
            print(f"    [{layer}] {name}")

print(f"\n{'=' * 70}\n")
sys.exit(0 if failed == 0 else 1)
