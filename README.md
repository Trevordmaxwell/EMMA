# Config Presets

Drop-in YAML fragments that can be merged into base configs via `scripts/run_training.py --preset ...`.
Each preset should contain only the keys you want to override.

## Available Presets
- `probes_full.yaml` — enables auxiliary cosine / InfoNCE losses, tighter read metrics, and stricter logging.
- `cpu_quick.yaml` — lowers epochs and train_size for fast iterations.
- `nce_read_guard.yaml` — requires read-alignment before InfoNCE opens, pauses λ when reads slip, and keeps a short post-gate oracle mix ramp (also enables extended per-epoch logging).

Add new presets here and document them.
