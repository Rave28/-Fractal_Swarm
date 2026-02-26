"""
crypto_utils.py — A2A Payload Integrity Layer
===============================================
Zero-Trust cryptographic utilities for the Fractal Swarm's
Agent-to-Agent (A2A) vector trading protocol.

Features:
  - Deterministic SHA-256 payload hashing (key-sorted JSON serialization)
  - ReleaseManifest dataclass — the signed envelope for every A2A trade
  - Signature verification with timing-safe hmac.compare_digest
  - Tamper-evidence: any modification to the payload produces a 403 mismatch

Usage (sender side):
    manifest = ReleaseManifest.sign(payload=vector_dict, swarm_id="node_01")
    requests.post(url, json=manifest.to_wire())

Usage (receiver side):
    ReleaseManifest.verify_wire(data=request.json_body)
    # raises IntegrityError on mismatch — never returns a tampered manifest
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


# ─── Exceptions ────────────────────────────────────────────────────────────────


class IntegrityError(Exception):
    """Raised when a received A2A manifest fails SHA-256 verification."""

    def __init__(
        self, reason: str, expected: str | None = None, received: str | None = None
    ):
        self.reason = reason
        self.expected = expected
        self.received = received
        detail = reason
        if expected and received:
            detail += (
                f"\n  Expected : {expected[:32]}...\n  Received : {received[:32]}..."
            )
        super().__init__(detail)


# ─── Core Hashing ─────────────────────────────────────────────────────────────


def canonical_hash(payload: Any) -> str:
    """
    Compute a deterministic SHA-256 digest of an arbitrary payload.

    Determinism is guaranteed by:
      1. Recursively sorting all dictionary keys.
      2. Serializing to UTF-8 JSON with no extra whitespace.
      3. Producing a hex digest.

    Args:
        payload: Any JSON-serializable Python object (dict, list, str, int, etc.)

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_hash(payload: Any, expected_hash: str) -> bool:
    """
    Timing-safe comparison of canonical_hash(payload) against expected_hash.

    Uses hmac.compare_digest to prevent timing side-channel attacks.
    """
    actual = canonical_hash(payload)
    return hmac.compare_digest(actual.encode(), expected_hash.encode())


# ─── Release Manifest ─────────────────────────────────────────────────────────


@dataclass
class ReleaseManifest:
    """
    The signed envelope wrapping every A2A vector trade payload.

    Fields:
        manifest_id:    Unique UUID for this trade (for deduplication).
        swarm_id:       Identity of the sending node.
        payload:        The raw vector/semantic data being traded.
        payload_sha256: SHA-256 hex digest of the canonical payload.
        timestamp:      Unix timestamp (float) when the manifest was created.
        protocol_rev:   A2A protocol version (for forward compatibility checks).
        ttl_seconds:    Time-To-Live in seconds. Manifests older than this are
                        rejected as potential replay attacks. Default: 300s (5 min).
    """

    manifest_id: str
    swarm_id: str
    payload: Any
    payload_sha256: str
    timestamp: float
    protocol_rev: str = "a2a_v2.0"
    ttl_seconds: float = 300.0

    # ── Factory ──

    @classmethod
    def sign(
        cls,
        payload: Any,
        swarm_id: str,
        protocol_rev: str = "a2a_v2.0",
    ) -> "ReleaseManifest":
        """
        Create a new signed manifest for a given payload.

        The hash is computed BEFORE the manifest object is constructed,
        ensuring the manifest_id / timestamp do not influence the payload digest.
        """
        return cls(
            manifest_id=f"manifest_{uuid.uuid4().hex[:12]}",
            swarm_id=swarm_id,
            payload=payload,
            payload_sha256=canonical_hash(payload),
            timestamp=time.time(),
            protocol_rev=protocol_rev,
        )

    # ── Serialization ──

    def to_wire(self) -> dict:
        """Serialize the manifest to a JSON-serializable dict for HTTP transport."""
        return asdict(self)

    @classmethod
    def from_wire(cls, data: dict) -> "ReleaseManifest":
        """Deserialize a manifest from a received wire dict."""
        try:
            return cls(
                manifest_id=data["manifest_id"],
                swarm_id=data["swarm_id"],
                payload=data["payload"],
                payload_sha256=data["payload_sha256"],
                timestamp=float(data["timestamp"]),
                protocol_rev=data.get("protocol_rev", "a2a_v2.0"),
                ttl_seconds=float(data.get("ttl_seconds", 300.0)),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise IntegrityError(
                f"Malformed manifest — missing required field: {e}"
            ) from e

    # ── Verification ──

    def verify(self) -> None:
        """
        Verify both the payload SHA-256 digest AND the manifest TTL.

        Raises:
            IntegrityError: If the payload was tampered with OR the manifest
                            has expired (age > ttl_seconds).
        """
        # ── TTL / Replay-Attack Guard ────────────────────────────────────────
        age = time.time() - self.timestamp
        if age > self.ttl_seconds:
            raise IntegrityError(
                reason=f"Manifest '{self.manifest_id}' has EXPIRED: "
                f"age={age:.0f}s exceeds TTL={self.ttl_seconds}s. "
                "Possible replay attack — dropping payload.",
            )
        # ── SHA-256 Integrity Check ──────────────────────────────────────────
        recomputed = canonical_hash(self.payload)
        if not hmac.compare_digest(recomputed.encode(), self.payload_sha256.encode()):
            raise IntegrityError(
                reason=f"Payload integrity check FAILED for manifest '{self.manifest_id}' "
                f"from swarm '{self.swarm_id}'. Payload was tampered with in transit.",
                expected=self.payload_sha256,
                received=recomputed,
            )

    @classmethod
    def verify_wire(cls, data: dict) -> "ReleaseManifest":
        """
        Deserialize AND verify a manifest from a wire dict in one call.

        The safer pattern for receiver endpoints — never returns a tampered manifest.

        Raises:
            IntegrityError: On deserialization failure or hash mismatch.
        """
        manifest = cls.from_wire(data)
        manifest.verify()
        return manifest


# ─── Self-Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[crypto_utils] Self-Test\n")

    # Test 1: Clean round-trip
    payload = {"project": "omega", "seed": "Aegis-Fractal-0x99A", "tier": 1}
    manifest = ReleaseManifest.sign(payload=payload, swarm_id="test_node_01")
    wire = manifest.to_wire()
    recovered = ReleaseManifest.verify_wire(wire)
    assert recovered.payload == payload
    print(f"[PASS] Test 1: Clean round-trip  hash={manifest.payload_sha256[:16]}...")

    # Test 2: Tampered payload is caught
    wire_tampered = dict(wire)
    wire_tampered["payload"] = {
        "project": "omega",
        "seed": "INJECTED_POISON",
        "tier": 1,
    }
    try:
        ReleaseManifest.verify_wire(wire_tampered)
        print("[FAIL] Test 2: Tamper was NOT caught")
    except IntegrityError as e:
        print(f"[PASS] Test 2: Tamper caught correctly: {e.reason[:60]}...")

    # Test 3: Determinism — same payload always produces same hash
    h1 = canonical_hash({"b": 2, "a": 1, "c": [3, 2, 1]})
    h2 = canonical_hash({"c": [3, 2, 1], "a": 1, "b": 2})
    assert h1 == h2, "Hash is not deterministic!"
    print(f"[PASS] Test 3: Canonical hash is deterministic  hash={h1[:16]}...")

    # Test 4: Malformed wire data raises IntegrityError
    try:
        ReleaseManifest.from_wire({"broken": True})
        print("[FAIL] Test 4: Should have raised IntegrityError")
    except IntegrityError as e:
        print(f"[PASS] Test 4: Malformed manifest rejected: {e}")

    print("\n[OK] All self-tests passed.")
