# Contributing

This workspace is tuned for quick iteration with continuity across users. To add features or experiments:

## 1. Environment
- Use Python 3.10+ and the shared virtualenv: `python3 -m venv .venv_emma && source .venv_emma/bin/activate`.
- Install deps: `pip install -r requirements.txt` (PyTorch CPU wheels install automatically in the run scripts).

## 2. Code Changes
- Source lives under `src/`. Keep new modules inside the `emma/` package unless they are utilities.
- Add configs in `configs/` and scripts in `scripts/` (only if they are reusable; one-offs belong in `legacy/`).
- Log experiments to `experiments/runs/` and summarize insights in `docs/progress.md`.

## 3. Tests
- Lightweight smoke tests live in `tests/`.
  - `pytest tests/test_memory_reset.py` ensures the VSA memory resets between forwards.
  - `pytest tests/test_train_smoke.py` covers a short training epoch on CPU.
- Add tests for new behaviors; keep them fast (≤10s) so contributors can run them routinely.

## 4. Documentation
- Update `docs/progress.md` with new findings or TODOs.
- Extend `docs/architecture.md` if you change the model/memory interface.
- Note any new scripts or workflows in `docs/README.md`.

## 5. Legacy
- Treat `legacy/` as read-only. If you revive an artifact, promote it into the active tree and document the move.

## 6. Pull Requests / Handoffs
- Before handing off, run `pytest tests` and include the latest experiment results (path + short summary).
- Mention any outstanding questions or follow-ups in `docs/progress.md`.

Thanks for keeping EMMA tidy and reproducible!
