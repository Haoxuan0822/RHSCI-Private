# π0.5 两桌评测：STATE 与 VGQ-AE

本目录存放本机实验编排，不属于正式训练/推理框架，不作为公开发布入口。数据、权重、环境和输出均在 `/root/workspace/Haoxuan/RHSCI-runtime`。正式推理调用 `rhsci.inference.http_server`，仿真直接调用官方 SIMPLE `run_eval`；没有复制模型或评测循环。

## 论文、README 与本次设置

启动前已阅读 [SIMPLE v2 论文附录 A.3](https://arxiv.org/html/2606.08278v2#A3)、[官方 π0.5 README](https://github.com/physical-superintelligence-lab/Psi0/blob/main/baselines/pi05/README.md)、[SIMPLE decoupled WBC 评测说明](https://github.com/physical-superintelligence-lab/SIMPLE/blob/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11/docs/source/decoupled-wbc/evaluation.md)，再核对保存的权重配置与实际实现。

| 项目 | 本次配置 | 来源与差异 |
| --- | --- | --- |
| 模型 | π0.5 STATE、VGQ-AE，各 checkpoint-40000 | HF RHSCI 已训练权重，不重新训练 |
| 任务 | G1WholebodyLocomotionPickBetweenTablesTeleop-v0 | 官方两桌 Teleop 任务 |
| 预测 / 执行 | 30 / 30，36 维动作，无 RTC | 用户明确指定；权重 horizon=30。论文 π0.5 chunk=16，官方 README 的 SIMPLE 示例 horizon=30，因此不称完全复现论文配方 |
| 采样 | 原生 flow matching，10 步，BF16 autocast | 当前 RHSCI π0.5 推理配方；不将未明确的论文设置推断为相同 |
| 输入 | 当前左目 RGB；STATE 另用 32 维状态；VGQ-AE 推理不读状态/历史 | 保存的模型方法及现有处理器 |
| 归一化 | 恢复训练时配置与匹配的数据统计；不替换为新版统计 | 保存的 contract、rsci_config、训练数据 fingerprint |
| 仿真 | MuJoCo 3.3.6 + Isaac Sim 4.5.0.0，headless | 官方 README 推荐组合；SIMPLE 固定 b75c6786 |
| 控制 | 官方 SONIC decoupled WBC，50 Hz；每块覆盖 0.6 秒仿真时间 | 官方 Teleop 评测说明及控制配置 |
| 初态 | SIMPLE-eval/fb67cd9a，level-0/1/2/hard，各 episode 0–9 | 最新评测资源；论文表格只有 level-0/1/2，hard 为更新后的扩展 |
| 次数 | 每个难度重复 5 轮 × 10 次；每模型 200 次，总计 400 次 | 用户指定；每模型只有 40 个不同初态，重复次数不作为独立初态数 |
| 时限 / 判定 | 1200 控制步；success_criteria=0.4 | 官方任务 metadata，并非从论文推断；保留官方稳定阶段最多 300 步 |
| 随机性 | 原生随机动作采样；状态测量噪声关闭 | 无 RTC；未修改官方客户端注入自定义确定性协议 |
| 输出 | 保存逐 episode 视频、官方统计、异常和每轮命令 | 每轮独立工作目录，避免覆盖 |

本次没有修补官方已知的 WBC/MuJoCo 手指映射问题，沿用固定官方实现，限制见 `docs/SIMPLE_UPDATES_20261103.md`。

## 8 卡分配与实际启动入口

| GPU | 模型 | 难度 | HTTP 端口 | DDS domain |
| --- | --- | --- | --- | --- |
| 0 | STATE | dr-level-0 | 8120 | 100 |
| 1 | STATE | dr-level-1 | 8121 | 101 |
| 2 | STATE | dr-level-2 | 8122 | 102 |
| 3 | STATE | dr-level-hard | 8123 | 103 |
| 4 | VGQ-AE | dr-level-0 | 8124 | 104 |
| 5 | VGQ-AE | dr-level-1 | 8125 | 105 |
| 6 | VGQ-AE | dr-level-2 | 8126 | 106 |
| 7 | VGQ-AE | dr-level-hard | 8127 | 107 |

准备完成后，在仓库根执行：

```bash
cd /root/workspace/RHSCI-Private
bash experiments/pi05_between/launch_8gpu.sh
```

脚本逐项显式指定模型、GPU、难度和 5 轮次数。每个 worker 先启动下面的原生服务并验证一次真实 `/act` 返回有限的 `(30, 36)`，再连续启动 5 个官方评测进程。预热不计入评测次数。

以下是 GPU 0 的实际服务命令；其他卡仅按表替换 GPU、run-dir 和端口：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
export CUDA_VISIBLE_DEVICES=0

python -m rhsci.inference.http_server \
  --backbone pi05 \
  --run-dir "$PSI_RUNTIME_ROOT/checkpoints/pi05-STATE-between-40000" \
  --checkpoint-step 40000 \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8120 \
  --no-deterministic-eval
```

预测/执行 30/30、10 步采样为现有服务的实际固定实现，不虚构不存在的 CLI 参数。VGQ-AE 的 run-dir 为 `$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1`。

GPU 0 第一轮的实际客户端命令如下。`eval_decoupled_wbc.py` 是对官方 `run_eval(EvalConfig(...))` 的薄封装，只额外暴露渲染卡和 DDS domain 参数，使 8 个进程隔离；其余参数与官方命令格式一致：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
export CUDA_VISIBLE_DEVICES=0
export task=G1WholebodyLocomotionPickBetweenTablesTeleop-v0
export result_root="$PSI_RUNTIME_ROOT/evals/pi05-between-30x30-5cycles-20261104"
cd "$result_root/STATE/dr-level-0/repeat-01"

python /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
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
  --max-episode-steps 1200 \
  --success-criteria 0.4 \
  --eval-dir "$PWD/stats" \
  --save-video \
  --renderer-gpu 0 \
  --dds-domain-id 100 \
  --cpu-threads 8
```

每一轮实际执行的绝对路径命令另存于输出目录的 `command.sh`；模型服务命令和 PID 写入 `worker.json`。官方客户端名 `psi0_decoupled_wbc` 表示所用 HTTP 消息协议，并非实际加载 Ψ0 模型。

8 实例并行时显式将每实例 Carbonite/TBB 工作线程限制为 8（Kit `threadCount` / `maxThreadCount`），依据 [NVIDIA 线程配置说明](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html) 处理本机默认每实例数百线程造成的 CPU 争用；这是运行资源配置，不是论文预测/执行设置。此前无上限并行尝试的日志保存在同名 `-before-thread-limit` 输出目录。

## 资源与进度

固定模型 revision `59a1ea14d4346496d4e55c9fa58ea2f3330fd298`，训练数据来源为用户指定 dataset revision `ea9d480ff1d95f34b63d7ea368b1bc72e353fd57`。下载完成核对 HF SHA256，再恢复模型配置、训练统计和 tokenizer；模型不需要 optimizer/RNG 即可推理。

运行根为 `/root/workspace/Haoxuan/RHSCI-runtime/evals/pi05-between-30x30-5cycles-20261104`。`worker-gpu*.log` 为编排日志，每卡目录有 `server.log`，每轮有 `console.log`、`stats/` 和视频。Kit 关闭可能提前退出 Python，因此每轮客户端退出后额外核验官方 `stats/eval_stats.txt` 中 episode 0–9 恰好各一条，再生成 `verified_results.json`；不以关闭之后可能未写出的 `result.json` 或退出码 0 代替。仅有 `warmup.json` 不能视为物理评测完成；每卡五轮均验证共 50 次后才写 `COMPLETE.json`。

## 材质与 USD 引用检查

2026-11-04 追加按官方 ZIP 原始清单复核：49 个场景包与 1 个材质包，共 34068 个文件，文件名集合缺少 0、多出 0、大小/ZIP CRC32 不符 0；88 处 props/textures 别名与其对应官方目录文件集合和内容一致。归档 SHA256 在恢复时已验证，本次文件内容比较为大小与 CRC32，不将其称为再次全量 SHA256。官方材质包本身含有嵌套目录，按原样保留。命令为 `python experiments/pi05_between/audit_material_inventory.py`，完整结果在运行根 `logs/material-inventory-audit.json`。

全 40 初态的实际 reset/render 与桌面 MDL 绑定检查现使用下列命令，结果逐初态写入 `logs/pi05-between-material-rendering-all40/progress.json`，全部通过后才生成 `result.json`；这是资源诊断，不计入正式评测。检查有效绑定的 MDL 路径和导出名称，不能把“MDL 文件存在”当作绑定正确。官方保存的 `ground_material` 字段在当前 DR 实现中没有应用，保持官方行为，没有自行额外绑定地面材质。

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
CUDA_VISIBLE_DEVICES=7 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/check_material_rendering.py \
  --all-episodes
```

```bash
python3 experiments/pi05_between/check_materials.py
```

已静态检查 40 个初态中的 80 处材质使用：63 个不同 MDL 文件及其导出材质名称、175 个不同纹理路径全部存在。收据在运行目录 `logs/pi05-between-material-paths.json`。此检查不替代实际 USD/MDL 加载；客户端开启 `SIMPLE_ISAAC_LOG_LEVEL=warning` 保留引用警告，检查各轮 `console.log`。

实际渲染进一步复现了 `/root/tmp/<编号>/props/*.usd` 缺失引用：场景含向上多层的相对路径，解析后命中指向旧目录的失效链接。`strings` 无法可靠读取 USD crate 的压缩引用；修复脚本使用 USD API 读取引用、解析路径，并检查/恢复 88 处 props 与 textures 兼容别名。两者必须同时恢复，否则 USD 组合可正常但纹理仍缺失。旧链接改名保留，官方 USD 不修改。命令如下，收据为 `logs/pi05-between-usd-aliases.json`：

```bash
uv run --python /root/miniforge/bin/python3 --with usd-core --no-project \
  python experiments/pi05_between/repair_usd_paths.py

source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
CUDA_VISIBLE_DEVICES=7 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/check_material_rendering.py
```

四难度各 episode 0 的真实 reset/render 诊断结果写入 `logs/pi05-between-material-rendering/result.json`，不计入评测次数；需同时检查 warning 日志与图像。恢复环境时也已修正 vMaterials_2 额外嵌套目录。篮子空碰撞 mesh 报错不属于材质缺失，保留该已知问题记录；具体错误证据与当前验证状态维护在 `docs/ERROR_LOG.md`。

## 单卡双路容量试跑

结果：第二路模型真实预热通过；Isaac 场景加载阶段 GPU 2 峰值 30336 MiB，余量降到 1774 MiB，触发既定 2 GiB 保护，第二路停止并清理，原来的 8 路继续。尚未完成这一个诊断 episode，因此不能宣称双路完整评测已通过；没有自动扩大到每卡两路。此结果表明当前两套独立服务/仿真组合的余量很紧，不是一次已经发生的 OOM。原生 Kit 明确记录约 300 秒的 RTX RtPso 着色器编译等待；未通过降画质、删材质来回避峰值。

用户要求检查单卡能否同时跑两套完整评测。原 8 路继续运行；在 GPU 2 额外启动独立 STATE 服务、Isaac、MuJoCo、WBC，端口 8130、DDS 110，输出 `evals/dual-gpu2-capacity-probe-20261104`。配置仍为 checkpoint-40000、30/30、无 RTC、1200 步、阈值 0.4、保存视频；只跑 level-0 的 1 个初态作为容量诊断，不计入正式每模型 200 次。原生流程与论文/README 对照不变。

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
cd /root/workspace/RHSCI-Private
python -u experiments/pi05_between/probe_dual_gpu.py
```

探针实际启动如下完整 worker，每两秒记录 `memory.jsonl`；余量低于 2048 MiB 时只停止第二路，并保留 `result.json`。不因 `/health` 或预热成功就扩展为全卡双路。

```bash
python -u experiments/pi05_between/run_evaluation.py \
  --gpu 2 --slot 1 \
  --variant STATE --level dr-level-0 \
  --num-repeats 1 --num-episodes 1 \
  --output-root /root/workspace/Haoxuan/RHSCI-runtime/evals/dual-gpu2-capacity-probe-20261104
```
