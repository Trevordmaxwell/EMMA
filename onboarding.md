# Onboarding

Welcome! The workspace is now organized around clear areas:
- Active code lives in `src/` with configs in `configs/`.
- Experiments (logs + metrics) live under `experiments/`.
- Documentation (including this page) is collected in `docs/`.
- Legacy bundles and large archives sit in `legacy/` for reference only.

## Getting Started
1. Skim `docs/README.md` and `docs/progress.md` to understand current focus.
2. Create/activate a venv: `python3 -m venv .venv_emma && source .venv_emma/bin/activate`.
3. Install deps: `pip install -r requirements.txt`.
4. Run your first experiment: `./scripts/run_cpu_256.command`.
5. Capture notes or findings back in `docs/progress.md` (append under Recent Changes).

## Repo Conventions
- Keep new experiments inside `experiments/runs/` and `experiments/results/` (scripts already do this).
- Prefer updating `docs/progress.md` and the relevant report before starting a new deep dive.
- Treat everything under `legacy/` as read-only unless you promote items into the active tree.
- Add new scripts to `scripts/` only if they are general-purpose; otherwise document them under `legacy/scripts`.

## Customizing Runs Quickly
- Use `python scripts/run_training.py --config configs/cpu_len256.yaml --preset configs/presets/cpu_quick.yaml --set train.lr=0.002` to tweak parameters without editing the base YAML.
- Presets live under `configs/presets/` and can be combined; overrides use dotted keys (e.g., `--set emma.deq_max_iter=6`).
- Results automatically land in `experiments/runs/<run_name>/` with metrics mirrored to `experiments/results/`.

## Next Stops
- `docs/codebase.md` for module-level notes.
- `docs/reports/EMMA_Framework_Report.md` for design + knob overview.
- `docs/architecture.md` to help extend / update the system diagram.
