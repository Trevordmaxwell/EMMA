# Audit Toolchain (Legacy Bundle)

The historical audit setup now lives in `legacy/audit_bundle/`.

## Contents
- `legacy/audit_bundle/` — original audit-ready repo (training scripts, configs, notebooks, results).
- `legacy/scripts/` — one-click `.command` launchers (`EMMA_Audit_RunAll`, stability sweeps, benchmarks).
- `notebooks/colab/` — Colab entry points (`EMMA_Colab_AllInOne.py`, quickstart notebook).

## Using the Bundle Today
1. Copy the relevant folder from `legacy/audit_bundle/` into a scratch area if you need to rerun it.
2. Execute the legacy `.command` scripts from `legacy/scripts/` (they still expect the old layout).
3. Move fresh results into `experiments/` and summarize findings in `docs/progress.md`.

## Knob Cheat Sheet (still relevant)
- Data length `L`: runtime/memory ~ linear in `L` (tokens per batch ≈ `B × L`).
- Epochs `E`: total steps ≈ `E × ceil(train_size/B)`; stop when val accuracy plateaus.
- Batch `B`: memory/step ∝ `B × L`; reduce if memory-limited; larger `B` smooths gradients.
- Warm-start: teacher-force writes for the first few epochs; keep a small mix floor for stability.

Recommended starting points:
- CPU quick runs: `configs/cpu_len256.yaml` (warm-start 2, DEQ≈6–8).
- Strict L=512 probes: `configs/cpu_len512_nceA.yaml` (expect slower convergence on CPU; consider GPU).
- ListOps-lite sanity: `configs/listops_lite_len512_cpu.yaml` via `scripts/run_cpu_listops_512.command`.
