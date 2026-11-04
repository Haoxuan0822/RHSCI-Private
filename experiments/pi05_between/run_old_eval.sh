#!/usr/bin/env bash
# 使用独立 VGQ-AE 服务，复测 11 月 2 日同步前的官方三个等级。
set -euo pipefail

level=${1:?level 0/1/2}
gpu=${2:?renderer GPU}
port=${3:?existing VGQ-AE port}
domain=${4:?isolated DDS domain}
case "$level" in 0|1|2) ;; *) exit 2 ;; esac

source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=8
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
result_root="$PSI_RUNTIME_ROOT/evals/pi05-vgq-official-pre1102-singlethread-20261104"
data_root="$PSI_RUNTIME_ROOT/eval-official-pre1102"
task=G1WholebodyLocomotionPickBetweenTablesTeleop-v0

# 在 Kit 启动前等待真实服务就绪，避免模型加载时间被误判为评测失败。
ready=false
for attempt in {1..900}; do
  if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
    ready=true
    break
  fi
  sleep 2
done
if [[ "$ready" != true ]]; then
  printf 'VGQ-AE server readiness timeout on port %s\n' "$port" >&2
  exit 1
fi

for round in 1 2 3 4 5; do
  printf -v round_name 'repeat-%02d' "$round"
  output="$result_root/dr-level-$level/$round_name"
  # 不覆盖已有结果；只允许独立、全新的评测目录。
  mkdir -p "$(dirname "$output")"
  mkdir "$output"
  cd "$output"
  python -u /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
    "simple/$task" \
    psi0_decoupled_wbc train \
    --data-format lerobot \
    --data-dir "$data_root/$task/dr-level-$level" \
    --sim-mode mujoco_isaac \
    --host 127.0.0.1 --port "$port" --headless \
    --num-episodes 10 --episode-start 0 \
    --max-episode-steps 1200 --success-criteria 0.4 \
    --eval-dir "$output/stats" --save-video \
    --evaluation-round "$round" --video-rounds 2 \
    --renderer-gpu "$gpu" --dds-domain-id "$domain" --cpu-threads 8 \
    > "$output/eval.log" 2>&1
  # Kit 的正常关闭也可能提前结束 Python；必须核对真实的10条结果。
  python - "$output" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/workspace/RHSCI-Private/experiments/pi05_between")
from summarize_results import read_stats

output = Path(sys.argv[1])
stats = read_stats(output / "stats/eval_stats.txt")
if set(stats) != set(range(10)):
    raise RuntimeError(f"Incomplete old-package evaluation: {output}: {stats}")
(output / "verified_results.json").write_text(
    json.dumps({"episodes": 10, "stats": stats}, indent=2) + "\n"
)
PY
done
