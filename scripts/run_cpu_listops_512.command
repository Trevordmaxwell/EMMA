#!/bin/bash
# Run EMMA (ListOps-lite) len=512 config on CPU from the reorganized workspace.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_BIN="python3"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi
VENV_DIR="$PROJECT_ROOT/.venv_emma"
if [ ! -d "$VENV_DIR" ]; then
  "$PY_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q -r "$PROJECT_ROOT/requirements.txt" || true
python - <<'PY'
try:
    import torch
    print('torch present')
except Exception:
    import sys, subprocess
    print('Installing torch (CPU)...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch'], check=False)
PY
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export EMMA_PROJECT_ROOT="$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/experiments/runs"
RESULT_DIR="$PROJECT_ROOT/experiments/results"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$LOG_DIR/configs"
LOG="$LOG_DIR/listops_lite_len512.log"
CSV="$LOG_DIR/listops_lite_len512_per_epoch.csv"
CFG="$PROJECT_ROOT/configs/listops_lite_len512_cpu.yaml"
SAVE_CFG="$LOG_DIR/configs/listops_lite_len512_cpu.yaml"
cd "$PROJECT_ROOT"
"$PY_BIN" -u -m emma.train \
  --config "$CFG" \
  --model emma_liquid \
  --device cpu \
  --logfile "$LOG" \
  --save-config "$SAVE_CFG" \
  --metrics-out "$CSV" || true
python - <<'PY'
import json, os, pathlib
project_root = pathlib.Path(os.environ['EMMA_PROJECT_ROOT'])
log_path = project_root / 'experiments' / 'runs' / 'listops_lite_len512.log'
metrics = {}
if log_path.exists():
    for line in log_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if line.startswith('METRIC '):
            try:
                key, value = line.split('METRIC ', 1)[1].split('=')
                metrics[key] = float(value)
            except Exception:
                pass
result_path = project_root / 'experiments' / 'results' / 'listops_lite_len512_metrics.json'
result_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
print(json.dumps(metrics, indent=2, sort_keys=True))
print(f'Saved metrics to {result_path}')
PY
