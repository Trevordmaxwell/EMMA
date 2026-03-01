# Onboarding

Welcome! The workspace is now organized around clear areas:
- Active code lives in the `emma/` package with configs in `configs/`.
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
- Keep new experiments inside `experiments/logs/` and `experiments/metrics/` (scripts already do this).
- Prefer updating `docs/progress.md` and the relevant report before starting a new deep dive.
- Treat everything under `legacy/` as read-only unless you promote items into the active tree.
- Add new scripts to `scripts/` only if they are general-purpose; otherwise document them under `legacy/scripts`.

## Customizing Runs Quickly
- Use `python scripts/run_training.py --config configs/cpu_len256.yaml --set train.lr=0.002` to tweak parameters without editing the base YAML.
- Organize overrides with dotted keys (e.g., `--set emma.deq_max_iter=6`) and commit final YAMLs back into `configs/` when stabilised.
- Results automatically land in `experiments/logs/<run_name>.log` with metrics mirrored to `experiments/metrics/` and plots to `experiments/plots/`.

## Next Stops
- `docs/codebase.md` for module-level notes.
- `docs/EMMA_Framework_Report.md` for design + knob overview.
- `docs/architecture.md` to help extend / update the system diagram.
