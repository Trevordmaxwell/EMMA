# Codebase Notes (`emma`)

## Package Layout
- `emma/model.py` — EMMA core model combining embedding, DEQ block, Liquid cell, and VSA memory.
- `emma/modules/` — lower-level building blocks (`deq_block.py`, `liquid_cell.py`, `vsa_memory.py`).
- `emma/data.py` — synthetic Needle-in-a-Haystack + ListOps-lite datasets.
- `emma/train.py` — config-driven trainer with warm-start/oracle + InfoNCE gating logic.
- `emma/utils.py` — device selection, seeding, InfoNCE helper.

## Running Locally
- Use `./scripts/run_cpu_256.command` for the standard Needle (L=256) benchmark.
- Use `./scripts/run_cpu_listops_512.command` for the ListOps-lite classification sanity check.
- Quick CPU presets: use the `configs/cpu_len*.yaml` variants that mirror the latest sweeps.
- For custom sweeps, run `python scripts/run_training.py --config <yaml> --set train.lr=0.002` to merge overrides and launch runs inside `experiments/logs/` with metrics in `experiments/metrics/` (supports `--dry-run`).
- Both scripts:
  - Create/share `.venv_emma` at the repo root.
  - Install deps from `requirements.txt` and ensure `torch` CPU wheel is available.
  - Set `PYTHONPATH="$PWD"` so the `emma` package resolves.

### Read sharpening knobs
- `read.sharpen_topk`: integer K; retains the top-K logits (others are downshifted by `read.sharpen_mask_margin`).
- `read.sharpen_temp`: temperature multiplier (<1.0 sharpens, >1.0 smooths) applied before CE.
- `read.sharpen_mask_margin`: margin subtracted from the per-batch minimum when masking (defaults to 5.0).
- `read.sharpen_mask_value`: explicit fill value (overrides margin logic).
- `read.sharpen_eval_only`: set `true` to apply sharpening only during evaluation.

### Stability knobs
- `emma.spectral_norm_keys`: apply spectral norm to the value/key projection heads (pairs with the DEQ residual spectral norm setting).
- `read.cleanup_mode`: `nearest` (default) averages top-k prototypes; `softmax`/`hopfield` applies softmax weighting for a learned cleanup.
- `read.cleanup_temp`: softmax temperature when cleanup mode is softmax.
  - Write logs to `experiments/logs/` and metrics/plots under `experiments/metrics` + `experiments/plots`.

## Key Training Knobs
Defined in `configs/*.yaml`:
- `data.length`, `data.n_pairs` — sequence length and number of needle pairs per sample.
- `emma.deq_max_iter` — iterations for the fixed-point solver.
- `emma.mem_into_deq`, `mem_scale` — inject memory read into DEQ update.
- `emma.warm_start_epochs`, `oracle_mix_*` — teacher forcing / mix schedule.
- `emma.nce_gate_require_read`, `nce_gate_read_patience`, `nce_lambda_pause_below_read` — opt-in gating guards to demand strong read cosine before opening InfoNCE and freeze λ if alignment drops.
- `emma.nce_gate_require_plateau` — require validation plateau in addition to read readiness before opening the gate.
- `emma.post_gate_write_cooldown_batches` — optional batch-level write throttle immediately after the gate opens (default 0 = no cooldown).
- `emma.oracle_mix_post_gate_*` — optional post-gate ramp that keeps partial oracle mixing for the first few open epochs.
- `emma.value_dropout` — dropout applied to the predicted write vector before normalization.
- `memory.write_norm_clip`, `memory.write_strength_schedule`, `memory.load_balance_coeff` — bound write magnitude, ramp write strength after the gate opens, and add a light load-balancing regulariser.
- `loss.lambda_*` — auxiliary losses (prediction CE, cosine alignment, InfoNCE).
- `loss.info_nce_lambda` — optional ramp schedule (start/end/max/epochs/ema) for λ_InfoNCE.
- `read.cleanup_blend` — blend raw and cleaned reads (e.g., 0.7 raw / 0.3 cleanup) to reduce cleanup crutch.

## Diagnostics Emitted
Trainer logs expose:
- `avg_fp_iters`, `residual_norm` — DEQ health.
- `write_cos`, `read_cos`, `read_cos_raw`, `read_cleanup_cos` — memory alignment metrics.
- `addr_entropy`, `write_bucket_entropy`, `writes_per_seq_*` — associative memory behavior.
- `read_cleanup_gap`, `bucket_collision_rate_{overall,mean,max}` — cleanup efficacy and bucket collision telemetry.
- `bucket_load_variance` — per-bucket load variance (useful when the load-balancing coefficient is active).

Refer to `docs/architecture.md` for diagrams and to `docs/progress.md` for current experiments.
