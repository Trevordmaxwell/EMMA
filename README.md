# EMMA Architecture (Minimal Release)

This folder contains a lean, GitHub‑ready snapshot of the EMMA architecture to accompany the paper (emma.pdf). It includes the core model, VSA memory, DEQ fixed‑point block, a tiny dataset, curated configs, and simple run scripts. Heavy artifacts and local experiments are omitted.

## What’s Included
- src/
  - data/needle.py — synthetic Needle‑in‑a‑Haystack task
  - modules/ — VSAMemory (HRR), FixedPointBlock (DEQ), LiquidCell
  - models/emma.py — EMMA model; models/baselines.py — GRU baseline
  - utils/ — device + seeding helpers
  - train.py — training loop with warm‑start + ramp scheduling and diagnostics
- configs/
  - needle_tiny.yaml — quick CPU sanity
  - cpu_n4_len512.yaml — mid‑length CPU recipe
  - colab_n8_len1024.yaml — tuned long‑context GPU recipe (n=8, L=1024)
  - colab_n8_len2048.yaml — extended long‑context recipe (CPU/MPS/GPU)
- scripts/
  - env.sh — sets env vars, PYTHONPATH
  - run_train.sh — wrapper to run training and tee logs
- tests/ — minimal sanity tests for VSA and DEQ
- requirements.txt, optional-requirements.txt, Makefile, .gitignore
- emma.pdf — paper to accompany this code

## Quickstart (CPU)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional extras; skip if they fail
pip install -r optional-requirements.txt || true

# Sanity tests
python -m tests.test_vsa
python -m tests.test_deq

# Tiny GRU baseline and EMMA
./scripts/run_train.sh configs/needle_tiny.yaml gru cpu
./scripts/run_train.sh configs/needle_tiny.yaml emma_liquid cpu
```

## Long‑Context Examples
- CPU mid‑length (n=4, L=512):
```bash
./scripts/run_train.sh configs/cpu_n4_len512.yaml emma_liquid cpu
```
- GPU/Colab (n=8, L=1024):
```bash
./scripts/run_train.sh configs/colab_n8_len1024.yaml emma_liquid cuda
```
- Extended (L=2048):
```bash
# CPU (safer on macOS if MPS is unstable at large L)
./scripts/run_train.sh configs/colab_n8_len2048.yaml emma_liquid cpu
# Or try MPS/GPU if available
./scripts/run_train.sh configs/colab_n8_len2048.yaml emma_liquid mps
./scripts/run_train.sh configs/colab_n8_len2048.yaml emma_liquid cuda
```

## Training Schedule & Diagnostics
- Warm start: teacher‑forced writes for `warm_start_epochs` (default: 2)
- Ramp: decay oracle→pred mix across `oracle_mix_ramp_epochs` to `oracle_mix_min`
- Memory injection: `mem_into_deq` true with scheduled `mem_scale`
- DEQ: `deq_max_iter`≈8; logs average fixed‑point iterations
- Logged per‑epoch: train/val loss, accuracy, `avg_write_cos`, `avg_read_cos`, top‑k hit rate

## Notes
- .gitignore excludes venv, logs, and results; safe to push to GitHub.
- For macOS, MPS may assert at very long sequences on some machines; use CPU if that occurs.
- The code reads and writes to `runs/` and `results/` inside this folder.

Have fun exploring EMMA!
