#!/usr/bin/env bash
# Lightweight queue for the main SDSD-Safety runs.
# Usage: GPU=0 QUEUE=sdsd bash scripts/run_main_queue.sh
set -uo pipefail

ROOT=/data4/sjj/catelogue/sdsd_safety
PY=/data4/sjj/catelogue/alignment/safety-component/.venv-v31/bin/python
CONFIG=configs/qwen3_4b.yaml
GPU=${GPU:-0}
QUEUE=${QUEUE:-sdsd}
MIN_FREE_MIB=${MIN_FREE_MIB:-30000}
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
mkdir -p logs runs

wait_for_memory() {
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if [[ -n "$free" && "$free" -ge "$MIN_FREE_MIB" ]]; then
      return 0
    fi
    echo "[queue] GPU $GPU free=${free:-unknown}MiB < ${MIN_FREE_MIB}MiB, waiting $(date '+%F %T')" | tee -a "logs/queue_gpu${GPU}.log"
    sleep 60
  done
}

run_train() {
  local run_id=$1; shift
  echo "[queue] START $run_id on GPU $GPU $(date '+%F %T')" | tee -a "logs/queue_gpu${GPU}.log"
  wait_for_memory
  "$PY" -m sdsd_safety.train --config "$CONFIG" --run-id "$run_id" "$@" \
    > "logs/${run_id}.log" 2>&1
  local rc=$?
  echo "[queue] END $run_id rc=$rc $(date '+%F %T')" | tee -a "logs/queue_gpu${GPU}.log"
  return $rc
}

run_eval() {
  local run_id=$1; shift
  echo "[queue] EVAL $run_id on GPU $GPU $(date '+%F %T')" | tee -a "logs/queue_gpu${GPU}.log"
  wait_for_memory
  "$PY" -m sdsd_safety.evaluate --config "$CONFIG" --phase generate --run-id "$run_id" "$@" \
    > "logs/eval_${run_id}.log" 2>&1
  local rc=$?
  echo "[queue] EVAL END $run_id rc=$rc $(date '+%F %T')" | tee -a "logs/queue_gpu${GPU}.log"
  return $rc
}

case "$QUEUE" in
  sdsd)
    run_train sdsd_self_k1_seed42 --override train.method=sdsd || exit 1
    run_train sdsd_external_k1_seed42 --override train.method=sdsd --override train.demo_source=external || exit 1
    run_train sdsd_self_k3_seed42 --override train.method=sdsd --override train.num_demos=3 || exit 1
    ;;
  baselines)
    run_eval base --adapter none --benchmarks harmbench,xstest,gsm8k || exit 1
    run_train sft_seed42 --override train.method=sft || exit 1
    run_train grpo_seed42 --override train.method=grpo || exit 1
    run_train opsa_seed42 --override train.method=opsa || exit 1
    ;;
  resume)
    run_train sft_seed42 --override train.method=sft || exit 1
    run_train grpo_seed42 --override train.method=grpo || exit 1
    run_train opsa_seed42 --override train.method=opsa || exit 1
    ;;
  grpo_opsa)
    run_train grpo_seed42 --override train.method=grpo || exit 1
    run_train opsa_seed42 --override train.method=opsa || exit 1
    ;;
  *)
    echo "unknown QUEUE=$QUEUE" >&2
    exit 2
    ;;
esac
