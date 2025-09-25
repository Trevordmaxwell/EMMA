# EMMA Workspace

Reorganized home for EMMA (Equilibrium + Memory + Minimal Liquid) research. The layout now separates live code, experiments, and archival assets so newcomers can get productive quickly and past work stays discoverable.

## Quickstart
- Python 3.10+ recommended (`python3 -m venv .venv_emma && source .venv_emma/bin/activate`).
- `pip install -r requirements.txt`
- Run a CPU sanity check: `./scripts/run_cpu_256.command`
- Explore additional configs under `configs/` and log results to `experiments/`.

## Repository Layout
- `src/` — EMMA package (`emma/` modules, trainer, MANIFEST).
- `configs/` — YAML configs for needle & ListOps-lite tasks.
- `scripts/` — maintained entrypoints (`run_cpu_256.command`, `run_cpu_listops_512.command`).
- `experiments/` — canonical runs (`runs/`), metrics (`results/`), audit artifacts.
- `docs/` — onboarding, architecture notes, reports, figures, paper candidates.
- `notebooks/` — Colab quickstarts and analysis notebooks.
- `tools/` — lightweight maintenance scripts (e.g., cleanup helpers).
- `legacy/` — archived bundles, old scripts, historical repos (kept read-only).

## Workflow Hints
1. Pick or duplicate a config from `configs/`, update metadata in `docs/progress.md`.
2. Launch runs via `scripts/`, capture outputs under `experiments/` (logs auto-write there).
3. Record findings in `docs/reports/` or append to the `docs/progress.md` changelog.
4. Use `legacy/` only for reference; do not mutate without copying into the active tree.

## Orientation Docs
Head to `docs/README.md` for a curated map of architecture notes, experiment summaries, and next steps.
