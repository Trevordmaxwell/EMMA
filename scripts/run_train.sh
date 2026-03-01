#!/usr/bin/env bash
set -euo pipefail
if [ -z "${VIRTUAL_ENV:-}" ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export MPLCONFIGDIR="$PWD/runs/.mplcache"

CFG="${1:?cfg yaml required}"
MODEL="${2:?model required (gru|emma_liquid)}"
DEV="${3:-mps}"
LOG="runs/$(basename "${CFG%.yaml}")_${MODEL}_${DEV}.log"
shift 3

PYBIN="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
"${PYBIN:-python}" -u -m src.train --config "$CFG" --model "$MODEL" --device "$DEV" "$@" | tee "$LOG"
