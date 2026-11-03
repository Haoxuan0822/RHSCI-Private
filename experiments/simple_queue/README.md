# 当前两桌之后的自动评测队列

2026-11-04 用户授权：等当前 STATE / VGQ-AE 两桌 400 次全部完成，再评测以下 7 组。每组沿用四难度各 10 初态 × 5 轮，200 次，共新增 1400 次；重复不计作独立初态。准备进程现在下载，队列只有在当前 8 路各 50 次有逐轮官方结果、资源退出且后续权重就绪后才启动。实验代码复用 `experiments/pi05_between/run_evaluation.py`、正式模型 HTTP 和官方 SIMPLE `run_eval`。

| 顺序 | 模型 / 任务 | 权重步数 | 预测 / 执行 | 时限 / success criteria |
| --- | --- | --- | --- | --- |
| 1 | π0.5 NO-STATE 两桌 | 40000 | 30 / 30，无 RTC | 1200 / 0.4 |
| 2 | GR00T STATE、VGQ-AE 开烤箱 | 20000 | 16 / 16，无 RTC | 1000 / 0.5 |
| 3 | GR00T STATE、VGQ-AE 交接 | 20000 | 16 / 16，无 RTC | 800 / 0.9 |
| 4 | GR00T STATE、VGQ-AE 开水龙头 | 20000 | 16 / 16，无 RTC | 1000 / 0.2 |

来源先比对 [SIMPLE 论文附录 A.3](https://arxiv.org/html/2606.08278v2#A3)、[官方 GR00T README](https://github.com/physical-superintelligence-lab/Psi0/blob/main/baselines/gr00t-n1.6/README.md) 及其部署入口，再核 RHSCI `baselines/gr00t/README.md` 与保存的契约。论文 GR00T 明确无 RTC、顺序预测，未在该段明确写 16；16 来自现有 README/原生部署和 RHSCI 有效动作长度。π0.5 的 30/30 是用户指定，与论文 π0.5 的 16 不同。任务时限/判定来自固定官方任务 metadata，交接使用其构造函数默认 0.9，不能直接沿用两桌 0.4。GR00T 输入当前帧，STATE 读取状态，VGQ-AE 不用推理历史/状态；采样及 processor 保留保存配方，不替换为 π0.5 的 10 步采样。

固定模型 HF `ds1x/RHSCI@59a1ea14d4346496d4e55c9fa58ea2f3330fd298`，确切归档路径见 `plan.py`。下载器验证 HF SHA256，再安全解包；恢复训练契约、原模型和 processor/统计。GR00T 推理不额外要求训练数据或 optimizer。π0.5 继续使用已匹配的两桌训练数据与 tokenizer。GR00T 环境 Torch 2.7.0+cu128 / Transformers 4.51.3 / FlashAttention 2.7.4.post1 已导入验证，实际 GPU 推理兼容要等真实预热，不用导入成功代替。

初始化权重与论文 SIMPLE 特定预训练路径不同、固定官方手指映射问题均保留披露，见现有审查文档。这里交接为 `G1WholebodyHandoverTeleop-v0`，不是有新版成功判定缺口的 BendHandover。

在仓库根分别启动 CPU 准备与等待队列；环境变量沿用 `activate.sh`：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05

/root/miniforge/bin/python3 -u experiments/simple_queue/prepare.py
python -u experiments/simple_queue/run.py
```

实际运行时二者为并行后台进程，日志在运行根 `logs/followup-preparation.log` 与 `logs/followup-queue-active.log`。准备不占 GPU。用户最新限定只下载所列评测模型；全量下载调度、自动重试器和无关权重传输均已停止，保留分片，准备结束也不恢复它们。复用已在进行的同一路径传输，不并发写同一个 `.parts`。

当前下载已恢复：日本 Y05/Y06 专用线路端口均未连通，用户最新要求优先保证下载、必要时换回。下载组已切香港 Y01，真实权重 64 KiB Range 验证正确返回 206，再核对身份并 SIGCONT 原准备 PID 267176、烤箱下载 PID 267262/267263，保留分片续传，不重建任务。π0.5 NO-STATE、GR00T 交接 STATE 已完整准备；水龙头 STATE 归档已下载校验，其他指定权重继续。当前 400 次评测及等待队列始终运行。详细路由、DNS/端口故障证据见必读工作/错误记录。

每个模型四难度分别占四张卡；GR00T 每任务 STATE 占 GPU 0–3，VGQ-AE 占 4–7。CPU 资源沿用每实例 8 个 Carbonite/TBB 工作线程。完整 worker 参数由 `plan.worker_command()` 声明，实际绝对命令另存每个模型目录的 `command-gpu*.sh`，每轮完整 SIMPLE 命令仍为 `repeat-*/command.sh`。

队列通过本机现有 `/root/bin/xiancun pause 360` 每小时续暂停空闲占卡程序，防止六小时自动恢复干扰长评测；没有杀生产训练或禁用其监督服务。队列结束后不再续期。

GR00T 开烤箱 STATE 的实际预定模型服务命令（固定 inference 别名由准备进程恢复；当前等待前一批完成，尚未启动此模型）：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh gr00t
CUDA_VISIBLE_DEVICES=0 python -m rhsci.inference.http_server \
  --backbone gr00t \
  --run-dir "$PSI_RUNTIME_ROOT/checkpoints/followup/gr00t-STATE-oven/inference" \
  --checkpoint-step 20000 \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8120 \
  --no-deterministic-eval
```

对应第一轮 SIMPLE 客户端命令为：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
export task=G1WholebodyOpenOvenTeleop-v0
export result_root="$PSI_RUNTIME_ROOT/evals/followup-pi05-gr00t-5cycles-20261104"
cd "$result_root/gr00t-STATE-oven/STATE/dr-level-0/repeat-01"

CUDA_VISIBLE_DEVICES=0 python \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/$task \
  psi0_decoupled_wbc \
  train \
  --data-format lerobot \
  --data-dir "$SIMPLE_EVAL_ROOT/$task/dr-level-0" \
  --sim-mode mujoco_isaac \
  --host 127.0.0.1 \
  --port 8120 \
  --headless \
  --num-episodes 10 \
  --episode-start 0 \
  --max-episode-steps 1000 \
  --success-criteria 0.5 \
  --eval-dir "$PWD/stats" \
  --save-video \
  --renderer-gpu 0 \
  --dds-domain-id 100 \
  --cpu-threads 8
```

交接/水龙头按表替换任务、初态、时限/判定；最终参数及路径由实际 `worker.json` 和 `command.sh` 确认。队列失败保留输出并停止，不把缺结果当作模型失败率，不跳过技术故障继续声称跑完。
