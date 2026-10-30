#!/usr/bin/env bash
# Reuse sjj's existing environment; never install torch/CUDA here.
set -euo pipefail
ROOT=${SDSD_ROOT:-/data4/sjj/catelogue/sdsd_safety}
PY=${SDSD_PYTHON:-/data4/sjj/catelogue/alignment/safety-component/.venv-v31/bin/python}
cd "$ROOT"
exec "$PY" -m sdsd_safety.queue --config configs/qwen3_4b.yaml \
  --gpu "${GPU:-0}" --group "${QUEUE:-repair}" "$@"
