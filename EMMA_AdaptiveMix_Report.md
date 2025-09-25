# EMMA Adaptive Mix & Confidence Gating Report

## Overview
- Enabled confidence-gated writes with per-epoch τ schedules and warm-start bypass.
- Added adaptive oracle_mix_alpha floor tied to EMA(write_cos) with optional soft-floor schedule (`oracle_mix_soft_floor`).
- Extended logging/aggregation to track `write_gate_ratio`, `lift_off_epoch`, mix telemetry, and mem→DEQ gate behavior.

## Code References
- `src/models/emma.py`: adaptive gating, mem→DEQ sigmoid gate, write metrics.
- `src/train.py`: adaptive mix schedule, mix telemetry logging, KPI emission.
- `aggregate_runs.py`, `run_stability_sweep.py`, `analyze_memory_health.py`: new metrics propagated to reports.
- Config example: `configs/exp_adaptive.yaml`.

## Experiments (L=512, n_pairs=2, CPU)
Confidence τ sweep (runs `exp_w*`):

| write τ | mem τ | mem_scale | Best Acc | Gate Ratio | Lift-off | Read Cos | Wall (s) | Notes |
|---------|-------|-----------|----------|------------|-----------|----------|----------|-------|
| 0.2 | 0.30 | 0.5 | 0.9375 | 0.333 | 1 | 0.824 | 97 | Stable post-warm |
| 0.2 | 0.35 | 0.5 | 0.9375 | 0.333 | 1 | 0.823 | 90 | " |
| 0.3 | 0.30 | 0.5 | 0.9375 | 0.333 | 1 | 0.824 | 103 | " |
| 0.4 | 0.30 | 0.5 | 0.9375 | 0.333 | 1 | 0.824 | 95 | " |
| 0.2/0.3 @ mem τ=0.25 | 0.5 | 0.9063 | 0.250 | 1 | 0.0 | 212–233 | Collapse after warm |

Dynamic mem→DEQ sweep with soft mix floor (runs `exp2_*`):

| k0 | k1 | Best Acc | Read Cos | Gate Ratio | Wall (s) |
|----|----|----------|----------|------------|----------|
| 0.4 | 0.8 | 0.9375 | 0.473 | 0.333 | 71.8 |
| 0.5 | 1.0 | 0.9375 | 0.594 | 0.333 | 77.4 |
| **0.6 | 0.8** | **0.9375** | **0.693** | **0.333** | **71.9** |
| 0.6 | 1.0 | 0.9375 | 0.693 | 0.333 | 78.5 |

> Recommendation: use `oracle_mix_k0=0.6`, `oracle_mix_k1=0.8`, soft floor `{high:0.15, low:0.05, read_cos_tau:0.8}`; keeps accuracy while minimizing walltime.

## Long-Context Probe (L=2048, strict)
- Config: `configs/_tmp_len2048_quick.yaml` (train_size=256, epochs=2, adaptive mix + gating).
- First epoch: val_acc 0.992, read_cos ≈0.99, gate ratio 1.0 (warm). Post-warm evaluation still running; job cancelled after ~23 min due to wall clock. Further tuning needed for practical runtime (suggest GPU run or reduce sequence length for CPU tests).

## Artifact Locations
- Logs & per-epoch CSVs: `runs/exp_w*`, `runs/exp2_*`, `runs/mixsoft_*`.
- Aggregated metrics: `results/runs_db.csv`, `results/memory_health_summary.csv`.
- Raw config snapshots: `configs/_tmp_*.yaml` (generated during sweeps).

## Next Steps
1. Roll mix soft-floor into stability sweeps (update `run_stability_sweep.py` grid).
2. Run full L=2048 strict test on GPU or with longer CPU budget to confirm behavior.
3. Investigate partial gating (blend predicted writes when gate fails) to push `write_gate_ratio` > 0.4.

