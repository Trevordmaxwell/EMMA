# Progress & Plan (Living Log)

_Last updated: 2025-09-24_

## Recent Changes
- Added InfoNCE guardrails: capped λ at 0.02 (`nce_lambda_end/max`), introduced `nce_lambda_freeze_epochs_after_open=1`, and wired `lr_warm_restart_factor=0.5`/`lr_warm_restart_epochs=2` so the gate opens gently; trainer now logs `gate_open_fraction` and honors the freeze via the NCEScheduler updates.
- Adjusted `cpu_len768_guard.yaml` so the guarded len768 preset now lowers the read gate threshold to 0.63, retargets the read EMA to 0.66, drops the plateau requirement, and reinstates a one-batch post-gate cooldown while keeping the λ ramp gentle (`nce_lambda_increment=0.001`). Next: rerun `len768_guard_guarded_n4_latest` and watch gate open fraction (goal ≈20–40%) before touching the λ ramp.
- Reran `len768_guard_guarded_n4_latest` with the tweaked guard: gate tripped on epoch 6 (read-ready) and stayed open for the final two epochs (~29% of the schedule). Pre-gate val acc peaked at 0.969 and the run finished at 0.875 with `post_gate_drop_epochs=1`; λ hit the planned 0.0298 ceiling while `nce_read_ema≈0.74` and collisions stayed ≤0.30. Writes collapsed once λ engaged (`write_cos_last≈0.008` on epoch 7), so consider capping `info_nce_lambda.max`≈0.02 and/or layering a short LR cooldown before the next sweep.
- Guardrails validated: rerun delivered `gate_open_fraction=0.286`, preserved best val acc (0.969→0.969), and held collisions ≤0.30 with λ capped at 0.02; epoch-6 raw read cos stayed 0.889 with writes intact before the freeze released, so proceed to longer sanity runs or trim `nce_lambda_increment` if residual spikes persist.
- Extended to 10 epochs (`len768_guard_guarded_n4_long` with early-stop guard): gate stayed in band (`gate_open_fraction=0.30`) but the run tripped the post-gate drop alarm at epoch 8 once the λ freeze lifted (`raw_read_cos≈0.66`, val_acc 0.922), so plan to retry with `info_nce_lambda.increment=0.0005` (or an extra cooldown epoch) before scaling length.
- Tried slower ramp (`len768_guard_guarded_n4_long_lambda0005`, `nce_lambda_increment=5e-4` override); gate dynamics unchanged (open fraction still 0.30) and the post-freeze drop reappeared at epoch 8 (val_acc 0.922, raw read cos 0.66). Next mitigation: add a second cooldown epoch and/or extend the LR warm-restart window before moving to len1024.
- Added second cooldown (`post_gate_write_cooldown_batches=2`) and lengthened LR restart window (3 epochs). Long run still tripped the drop guard at epoch 8 (`len768_guard_guarded_n4_long_cooldown2`), so the next knob to try is extending `nce_lambda_freeze_epochs_after_open` or trimming the post-gate ramp before cloning to len1024.
- Restructured workspace into `emma/`, `configs/`, `scripts/`, `experiments/`, `docs/`, and `legacy/`.
- Added opt-in gating safeguards (`nce_gate_require_read`, `nce_gate_read_patience`, `nce_lambda_pause_below_read`) and post-gate oracle mix ramp controls so quick sweeps stay stable while new profiles can demand read readiness.
- Extended telemetry capture behind `logging.per_epoch_extended` to log train-time read/write cosine spread and bucket coverage in `per_epoch.csv`.
- Added plateau-aware NCE gating (`nce_gate_require_plateau`), collision telemetry, and per-write HRR renorm so throttled runs stay stable when the gate opens.
- Landed len768 guard config (`configs/cpu_len768_guard.yaml`) with conservative writes (α=0.99, β=0.5) and ran a 7-epoch sweep: gate opened at epoch 7 with λ=0.001; `train_read_ema≈0.745`, `write_cos_ema≈0.407`, and the raw→clean gap stayed ≈0.02 while post-gate val acc settled at 0.938 (Δ=-0.031 vs pre-gate).
- Tested a follow-up len768 run with post-gate write cooldown + lower write strength (β=0.3). Gate still opened at epoch 7, but val accuracy slid to 0.922 (Δ=-0.047) and read gap grew to ~0.033, so the extra throttling hurt retention; visuals saved under `~/Desktop/len768_guard_cooldown_visuals/` for comparison.
- Raised the read threshold to 0.70 while keeping cooldown=1 (write_strength back to 0.5); gate never opened across 7 epochs, val acc stayed at 0.969 with λ frozen at 0, demonstrating the guard can hold until we explicitly relax the threshold (plots in `~/Desktop/len768_guard_cooldown_ws05_tight_visuals/`).
- Relaxed the guard slightly (`threshold=0.65`, λ increment 0.001) with dropout/cleanup blending. Gate now opens cleanly on the len768 preset (n_pairs=2) once reads clear ~0.70; val acc settles ~0.938 with tight collision bounds (run `len768_guard_guarded`).
- Guarded len768 with heavier write load (`n_pairs=4`, run `len768_guard_guarded_n4_latest`): λ ramped to ~0.03 but the gate stayed closed; val acc held 0.969 through epoch 6 before sliding to 0.875 on epoch 7, so we’ll need either a lower threshold or plateau relax before scaling needles further.
- Relaxed the guard slightly (`nce_gate_read_threshold=0.68`, `nce_lambda_increment=0.001`, cooldown=1, β=0.5); gate opened at epoch 7, val acc finished at 0.938 (same Δ≈‑0.031 as baseline) and raw→clean gap stayed ~0.02 (`len768_guard_guarded`).
- Consolidated active training scripts under `scripts/` with shared venv + `PYTHONPATH` handling.
- Archived audit bundles, desktop launchers, and older repos into `legacy/` for reference.
- Centralized documentation index (`docs/README.md`) and migrated reports into `docs/`.
- Documented the architecture flow and references in `docs/architecture.md`; regenerated Graphviz/LaTeX assets under `docs/diagrams/`.
- Extracted InfoNCE gating/oracle mix policy into `NCEScheduler` (`emma/schedules.py`) and simplified the trainer.
- Added smoke tests for memory reset, training loop, and scheduler behavior (`tests/`).
- Introduced `scripts/run_training.py` for config overrides + run logging automation.
- Validated the new launcher with `demo_cpu_len256_quick` (one epoch, probes preset); metrics saved under `experiments/metrics/demo_cpu_len256_quick_metrics.json`.
- Added read sharpening (top-k + temperature) controllable via `read.sharpen_*` config knobs.
- Introduced `emma.spectral_norm_keys` to stabilize key/value heads alongside existing DEQ spectral norm.
- Added optional softmax-based cleanup (`read.cleanup_mode=softmax`, `read.cleanup_temp`) for prototype snapping.
- CPU sweeps via `run_training.py` (quick preset + cleanup):
  - Len256: softmax-only (`val_acc=0.859`, loss 2.32) vs. top-2 (`0.859`, 1.21) / top-4 (`0.859`, 1.33`).
  - Len512: baseline (`val_acc=0.906`, loss 2.31) vs. top-2 (`0.906`, 1.20) / top-4 (`0.906`, 1.15`).
  - Len1024: baseline (`val_acc=0.906`, loss 2.26) vs. top-2 (`0.906`, 0.84) / top-4 (`0.906`, 0.94`).
  Metrics in `experiments/metrics/cpu_len256_softmax_*_metrics.json`, `experiments/metrics/cpu_len512_softmax_*_metrics.json`, and `experiments/metrics/cpu_len1024_softmax_*_metrics.json`.
- Added `scripts/run_adaptive_lambda_sweep.py` (quickstart-backed λ grid) plus a len512 quickstart preset; targeted adaptive sweeps now log under `experiments/logs/adaptive_*`. Len256 quick runs kept λ closed (short warmup), while len512 opened at epoch 3 with λ≈0.003 but slipped post-gate (val 0.906→0.859, loss 1.20).
- Extended the len256 sweep to 5 epochs with a shortened warm-start; λ opened by epoch 3 (≈0.003→0.005) but val loss ballooned to ~3.37 as read_ema hit 0.47. Completed the remaining len512/len1024 grid points: len512 still opens at epoch 3 and falls back to 0.859 post-gate, while len1024 never leaves warm-start and matches the softmax baseline.
- Len256 tune: four-epoch run with λ capped at 1e-3 kept the gate closed until epoch 4, held best acc 0.875, and dropped final loss to ~1.08 (read_ema ≈0.51).
- Len512 quickstart update: delaying the gate (start≥5) and trimming to two epochs preserves the 0.906 acc / 0.84 loss baseline while skipping the post-open slump; reserve λ sweeps for longer schedules.
- Added `--profile` bundles to `scripts/run_adaptive_lambda_sweep.py` (`len256:tuned`, `len512:no_gate`, `len512:small_lambda`) so the tuned schedules are one-flag reproducible. Longer len512 λ attempts still drop from 0.906→0.828 even with λ≤2e-3, so current guidance is to keep the gate closed for quick sweeps and log full-length retries separately.
  - Late-gate profile across seeds (42/43/44) yields best acc {0.906, 0.922, 0.938} with mixed final losses {0.829, 1.43, 0.828}; low λ ramp keeps read EMA high (~0.73) but seed 43 still regresses after gate (Δ−3pp), so flag variability in follow-up sweeps.
  - New `len512:late_gate` profile keeps λ off for six epochs, forces a low λ open at epoch 7, and held 0.906 acc with final loss 0.829 (nce_read_ema ≈0.74).
  - Tried `len512:cleanup_high` (higher oracle mix + tighter cleanup); gate waited until epoch 5 but accuracy still slipped to 0.844 once λ reached 1e-3, so profile is logged for experiments but not recommended for quick sweeps.
  - Sample telemetry (`tools/analyze_samples.py`) on seed 43 shows per-sequence writes collapsing (cos≈0.03) by epoch 3 and read cos falling to ≈0.83 at epoch 7 despite high global EMA, so the `len512:late_gate` profile now enforces `nce_gate_require_read=true` with a two-epoch patience before opening; seed 44 retains 0.93 acc while seed 43 still dips in the final epoch, so additional mitigation is needed.

## Stable Baselines
| Config | Task | Notes |
| --- | --- | --- |
| `cpu_len256.yaml` | Needle, L=256 | 6 epochs, DEQ=8, warm-start 2 epochs, InfoNCE gate optional.
| `cpu_len512_nceA.yaml` | Needle, L=512 | Strict micro-runs; requires patience on CPU, best with gating.
| `listops_lite_len512_cpu.yaml` | ListOps-lite | Uses `forward_classify`, warm-start + mix floor for stability.

Metrics for historical sweeps live under `experiments/metrics/`; raw logs in `experiments/logs/` with plots under `experiments/plots/`.

## Open Questions / Next Steps
1. Regenerate architecture diagrams (`emma_architecture_graphviz_v3.dot` → PDFs) to include scheduler abstractions.
2. Evaluate GPU support: prototype a no-grad GPU memory path or document the decision to remain CPU-only.
3. Automate smoke tests (CI or local pre-commit) so contributors run them by default.
4. Curate `legacy/` — identify bundles worth promoting (e.g., `legacy/audit_bundle`, `legacy/starter_kit`).
5. Extend scheduler coverage to integration tests (end-to-end run asserting lambda telemetry) once CI is in place.

Update this file whenever you run a new experiment or answer one of the open questions.
