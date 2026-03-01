# Colab Quickstart: EMMA vs Transformer (ListOps)

This guide runs the paired ListOps sweep in Google Colab with resume support.

## 1. Open the notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Click `File` -> `Open notebook`.
3. Use the `GitHub` tab and enter:
   - `https://github.com/Trevordmaxwell/EMMA`
4. Open `notebooks/EMMA_Colab_ListOps_Sweep.ipynb`.

If GitHub lookup is flaky, upload the notebook file manually from this repo.

## 2. Set runtime to GPU

1. In Colab, click `Runtime` -> `Change runtime type`.
2. Set `Hardware accelerator` to `GPU`.
3. Click `Save`.

## 3. Run cells top-to-bottom

Run each cell in order:

1. Mount Google Drive.
2. Review `Settings` and adjust:
   - `BRANCH_OR_SHA` (your branch or commit)
   - `SEEDS` (for example `60,61,62,63,64`)
   - `TRAIN_SIZE`, `VAL_SIZE`, `EPOCHS`
   - `RUN_PREFIX`
3. Clone + checkout + pull cached `experiments/` from Drive.
4. Install dependencies + preflight checks.
5. (Optional) Run smoke test.
6. Run full paired sweep.
7. Sync back to Drive and print summary metrics.

## 4. Where outputs live

Inside Colab repo clone:

- `experiments/runs/<run_name>/metrics.json`
- `experiments/results/<RUN_PREFIX>_summary.json`

In Google Drive cache:

- `MyDrive/EMMA_colab_cache/experiments/...`

## 5. Resume after disconnect

Rerun from the top and keep `--skip-existing` enabled. Completed seeds are skipped.

## 6. Common fixes

- `Missing required files` during preflight:
  - Set `BRANCH_OR_SHA` to a branch/commit that includes the new scripts.
- `CUDA not available`:
  - Re-check runtime type is GPU.
- `git checkout` fails:
  - Branch name typo; use exact branch or commit SHA.
- Slow or interrupted runs:
  - Reduce `TRAIN_SIZE` or `EPOCHS`, then scale back up after a successful pass.
