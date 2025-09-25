# EMMA Workspace

Reorganized home for EMMA (Equilibrium + Memory + Minimal Liquid) research. The layout now separates live code, experiments, and archival assets so newcomers can get productive quickly and past work stays discoverable.

## Quickstart
- Python 3.10+ recommended (`python3 -m venv .venv_emma && source .venv_emma/bin/activate`).
- `pip install -r requirements.txt`
- Run a CPU sanity check: `./scripts/run_cpu_256.command`
- Explore additional configs under `configs/` and log results to `experiments/`.

## Repository Layout
- `emma/` — importable package with model code, data utilities, schedulers, and training loops.
- `configs/` — YAML experiment definitions for needle, ListOps-lite, and CPU quick runs.
- `scripts/` — maintained entrypoints (`run_cpu_256.command`, `run_training.py`, etc.).
- `experiments/` — results organized into `metrics/`, `plots/`, and `logs/` for reproducibility.
- `docs/` — onboarding, architecture notes, long-form reports, and diagram source files.
- `notebooks/` — Colab quickstarts and exploratory analysis notebooks.
- `tests/` — pytest suite covering DEQ blocks, schedules, memory reset, and smoke training loops.
- `tools/` — lightweight maintenance helpers (e.g., evaluation placeholders).
- `legacy/` — archived scripts and historical implementations kept read-only.
- `EMMA_large_assets/` — oversized bundles preserved outside the main history.

## Workflow Hints
1. Pick or duplicate a config from `configs/`, update metadata in `docs/progress.md`.
2. Launch runs via `scripts/`, capture outputs under `experiments/` (logs auto-write there).
3. Record findings in `docs/reports/` or append to the `docs/progress.md` changelog.
4. Use `legacy/` only for reference; do not mutate without copying into the active tree.

## Orientation Docs
Head to `docs/README.md` for a curated map of architecture notes, experiment summaries, and next steps.
