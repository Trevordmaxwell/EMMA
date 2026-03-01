# Audit Toolchain Snapshot

Historical CPU audit bundle now lives at `legacy/audit_bundle/`.
- Contains training scripts, configs, Colab helpers, and archived results.
- One-click .command launchers moved to `legacy/scripts/`.

Highlights from previous sweeps:
- L=256 assisted runs stabilize after warm-start with DEQ≈6–8 iterations.
- Strict L=512 micro-runs can reach ≥0.92 post-warm with small `n_pairs`.
- Address entropy + DEQ residual metrics are the fastest indicators of collapse.

For new audits, copy relevant pieces into `scripts/` / `experiments/` and record outcomes in `docs/progress.md`.
