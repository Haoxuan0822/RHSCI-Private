#!/usr/bin/env bash
set -euo pipefail

source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
result_root="$PSI_RUNTIME_ROOT/evals/pi05-between-30x30-5cycles-20261104"
mkdir -p "$result_root"
worker_pids=()

stop_workers() {
  for worker_pid in "${worker_pids[@]}"; do
    kill -TERM "$worker_pid" 2>/dev/null || true
  done
  wait || true
}
trap 'stop_workers; exit 130' INT
trap 'stop_workers; exit 143' TERM

launch_worker() {
  local gpu="$1" variant="$2" level="$3"
  python -u experiments/pi05_between/run_evaluation.py \
    --gpu "$gpu" \
    --variant "$variant" \
    --level "$level" \
    --num-repeats 5 \
    > "$result_root/worker-gpu${gpu}.log" 2>&1 &
  worker_pids+=("$!")
}

launch_worker 0 STATE dr-level-0
launch_worker 1 STATE dr-level-1
launch_worker 2 STATE dr-level-2
launch_worker 3 STATE dr-level-hard
launch_worker 4 VGQ-AE dr-level-0
launch_worker 5 VGQ-AE dr-level-1
launch_worker 6 VGQ-AE dr-level-2
launch_worker 7 VGQ-AE dr-level-hard

status=0
for worker_pid in "${worker_pids[@]}"; do
  wait "$worker_pid" || status=1
done
exit "$status"
