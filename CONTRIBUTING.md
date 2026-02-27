# 🤝 Contributing to Fractal Swarm

The Fractal Swarm is a production-grade, zero-trust autonomous system.
We welcome contributions, but all pull requests must adhere to the
strict mathematical and cryptographic boundaries of the
**AEGIS Governance Protocol**.

## 🛑 The Golden Rules

1. **No Unbounded Memory:** Any updates to the ingestion pipeline
   or `chunk_intelligence` must respect the `force_split` byte-boundary
   fallback. Out-of-Memory (OOM) vulnerabilities are an immediate PR rejection.
2. **AST Guard Supremacy:** Do not attempt to bypass
   `swarm_boundary_guard.py`. If your feature requires the nanobots to
   execute side-effects (e.g., shell commands, filesystem deletion),
   you must implement a formal `SwarmCapabilityToken` workflow.
3. **Zero Semantic Pollution:** If you are adding new tools to the
   Oracle or DeepCode MCP, ensure they rigorously pass the `workspace_id`
   through the ChromaDB `where` filters.

## 🧪 Testing Your Changes

Before submitting a PR, you must run the full v2.0 stress harness.
**Your PR must score a perfect 44/44.**

```powershell
# Run the certification harness
python pipelines/sandboxes/swarm_full_stress_v2.py
```

## Pull Request Checklist

- [ ] My code passes the `swarm_full_stress_v2.py` suite (44/44).
- [ ] I have not introduced any unbounded text-splitting logic.
- [ ] New A2A payload types are properly wrapped in a ReleaseManifest
      with SHA-256 signatures and a 300s TTL.
- [ ] I have updated the `swarm_threats.yaml` if my feature introduces
      a new attack vector.

Welcome to the Hive.
