# Architecture Notes

This document captures the current EMMA (Equilibrium + Memory + Minimal Liquid) dataflow and supporting components.

## Visual References
- `docs/diagrams/emma_architecture_graphviz_v3.pdf` — authoritative block diagram regenerated via Graphviz (`docs/diagrams/emma_architecture_graphviz_v3.dot`).
- `docs/diagrams/emma_architecture_v3.pdf` — LaTeX render suitable for publications (`docs/diagrams/emma_architecture_v3.tex`).
- Regenerate with `dot -Tpdf docs/diagrams/emma_architecture_graphviz_v3.dot -o docs/diagrams/emma_architecture_graphviz_v3.pdf` (add `-Tpng` for previews). If LaTeX is available, rebuild `emma_architecture_v3.pdf` using `pdflatex` and place outputs back into `docs/diagrams/`.

## Core Pipeline
1. **Embedding**: tokens → learned embeddings.
2. **Fixed-Point Block (DEQ)**: `ResidualUpdate` + `FixedPointBlock` iteratively refine latent state `z` using the current token embedding and optional memory injection.
3. **Liquid Cell Backbone**: the continuous-time RNN cell (`LiquidCell`) mixes `z` into hidden state `h` for temporal continuity.
4. **Write Path**: concatenate `z`/`h`, project via `z_to_value`, optionally mix with oracle targets, then write (under `torch.no_grad()`) into the VSA memory.
5. **Read Path**: at query positions the memory is read, optionally cleaned (`_cleanup_read`), and compared against value prototypes to produce logits.

## Memory Subsystem
- `VSAMemory` provides bucketed HRR binding/unbinding with decay, top-k selection, entropy metrics, and write/read traces for diagnostics.
- Writes now renormalize the touched slots (per-bucket L2) and record collision rates so late-gate schedules can monitor bucket hygiene.
- Predicted write vectors can be value-dropout perturbed and norm-clipped before binding, and a light load-balancing penalty keeps bucket occupancy variance in check.
- Memory is reset per forward pass; training losses operate on differentiable predictions before writes occur.
- Key metrics: `write_cos`, `read_cos`, bucket entropy, write coverage, read cleanup cosine.

## Training Loop Highlights
- `emma/train.py` orchestrates training via YAML configs, supporting auxiliary CE, cosine, InfoNCE losses, and optional regularizers.
- **NCEScheduler** (`emma/schedules.py`) owns InfoNCE gate/oracle-hold policy; each epoch the trainer requests a decision (lambda value, gate state, plateau/read readiness). Optional knobs can now require sustained read readiness before opening and freeze λ back to the floor when alignment slips.
- Warm-start teacher forcing, oracle mix schedules, and memory injection ramps execute after consulting the scheduler.
- Tripwires (read/write cosine minima, residual norms, val-acc minimum) remain for safety and early exits.

## Diagnostics & Telemetry
- DEQ health: `avg_fp_iters`, `residual_norm`.
- Memory health: `write_cos`, `read_cos`, `read_cos_raw`, `read_cleanup_cos`, bucket entropy, write bucket entropy, writes-per-seq.
- Scheduler telemetry: `lambda_nce_current_epoch`, `gate_state_before/after`, `nce_open_steps`, `nce_read_ema_cos`, `nce_gate_epoch`.
- Logs live under `experiments/logs/…`; metrics under `experiments/metrics/` (plots in `experiments/plots/`).

## Outstanding Work
- Keep diagrams aligned with code — regenerate PDFs when architecture changes (new modules, GPU paths, etc.).
- Document future extensions here (e.g., alternative schedulers, GPU memory backends).
- Consider embedding quick ascii diagrams or linking thumbnails for faster orientation.
