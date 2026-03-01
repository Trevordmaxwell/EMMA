# EMMA Framework Report (living document)

Purpose: capture design, knobs, diagnostics, experiments, and plans. Update continuously.

## Design
- EMMA: VSA memory + DEQ solver + streaming backbone (Liquid)
- Query read → logits over value prototypes; training with warm-start + mixing floor

## Knobs (YAML)
- Oracle: warm_start_epochs, oracle_mix_min, oracle_mix_ramp_epochs, oracle_mix_schedule
- Oracle post-gate: oracle_mix_post_gate_epochs, oracle_mix_post_gate_min, oracle_mix_post_gate_start
- Gate safety: nce_gate_require_read, nce_gate_require_plateau, nce_gate_read_patience, post_gate_write_cooldown_batches
- Memory: mem_dim, n_slots, k_top, mem_into_deq, mem_scale, decay (if configured)
- Solver: deq_max_iter (train/eval), tolerance; (optional) distill.teacher_deq
- Losses: lambda_pred_ce, lambda_write_cos, lambda_write_nce
- Regularizers: reg.addr_entropy_weight
- Distillation: distill.enable, weight, temp

## Diagnostics
- avg_fp_iters, residual_norm
- addr_entropy, read_cos, write_cos
- plateau_epoch (from per-epoch val_acc)
- train_read/write distribution (min/max/std) via optional `logging.per_epoch_extended`
- read_cleanup_gap, bucket_collision_rate_{overall,mean,max}

## Experiments
- L=256 assisted: stable ≥0.88 post-warm; deq≈6; mem_into_deq helpful
- L=512 strict micro-runs: ≥0.92 post-warm with small N
- ListOps-lite classification (CPU): functional with warm-start + mixing floor

## Gaps / Next (see gap-scan)
- Add addressing supervision (KL to teacher/soft targets)
- ECC + locality in memory
- Adaptive DEQ policy
- Capacity figure; constant-memory figure

## References
- VSA / HRR, DEQ fixed-point methods, SSM/Liquid cells

## Changelog
- yyyy-mm-dd: Initial consolidation; added sweep and aggregator
