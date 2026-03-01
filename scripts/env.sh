#!/usr/bin/env bash
set -euo pipefail
if [ -z "${VIRTUAL_ENV:-}" ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export MPLCONFIGDIR="$PWD/runs/.mplcache"
