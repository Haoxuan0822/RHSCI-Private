# 当前两桌之后的自动评测队列

## 当前执行范围：只测GR00T与官方旧包（2026-11-04最新用户指令）

当前第四次队列2329010/session37378，日志`logs/gr00t-old-package-queue-usd-compat.log`：六路warmup有限(16,36)，Kit后各修复79处引用，开始原生episode初始化/双目录像；当前缺失USD/纹理/UsdToMdl/Traceback均0，暂无完成episode，不宣称完整验收。第三次在Kit后遇到USD版本API差异（GetAppliedItems不存在），已保留全部输出为`*-before-usd-api-compat`；改为计数原始六类list-op项目，CPU三场景再验组合错误0，收据`logs/gr00t-old-scene-reference-audit-listop/result.json`。前三次技术中止均不计模型失败，下列PID为历史。

最新第三次启动：队列2321277、工具session97966，日志`logs/gr00t-old-package-queue-kit-hook.log`，六worker2321278–2321283。二次2314573因Kit尚未初始化就导入pxr而退出；其目录完整移到同级`*-before-kit-hook`，不计模型失败。现在使用实验包装器先执行官方`BaseDualSim._init_isaac`再修正引用，保留原生预加载/启动顺序，随后才加载场景。实际渲染核验尚未完成，不能仅据六worker启动宣称通过。

首次队列PID2292579/工具session52445已因新场景USD错误停止：六路warmup有限(16,36)，但初始化102344280时出现运行根/props材质引用缺失，不计正式评测。完整目录已移到同级gr00t-official-pre1102-5cycles-20261104-before-oven-usd-repair，可恢复，未删除。修复后沿下方同一调度命令、全新输出目录重启；实际状态以新PID/日志、各轮stats及verified_results/COMPLETE为准。

本次启用显式--repair-scene-references：scene_references.py在Kit已初始化、原生场景加载前，仅对当前初态scene name对应的USD Layer作进程内UpdateExternalReference，不Save，不改磁盘官方USD/props/纹理。三场景共有308个失效路径（79/27/202），均指向运行根props；全场景扫描发现75个同名不同内容冲突，禁止建立共用props别名。两处仅文件名大小写不符：交接black.usd→Black.usd，文件内原材质严格为/World/Looks/black；水龙头White.usd→WHITE.usd，文件内原材质严格为/World/Looks/White。仅唯一casefold候选且准确材质prim存在时允许；不生成替代材质。保留原引用位置总数（不是去重文件数）、磁盘原USD SHA及每个目标props SHA，逐轮记录scene-reference-repair.json。CPU原生USD组合已通过三场景（5132/1758/9935 prim，组合错误0），日志gr00t-old-scene-reference-audit-retry/result.json；首次大小写预检拒绝日志/目录保留，不能当成功。原场景ZIP逐文件清单/大小/CRC32也复核，实际Isaac渲染验证随重启进行，不用CPU组合代替材质加载验收。

CPU复算命令：

```bash
uv run --python /root/miniforge/bin/python3 --with usd-core --no-project \
  python experiments/pi05_between/scene_references.py \
  --data-dir \
    /root/workspace/Haoxuan/RHSCI-runtime/eval-official-pre1102/G1WholebodyOpenOvenTeleop-v0/dr-level-0 \
    /root/workspace/Haoxuan/RHSCI-runtime/eval-official-pre1102/G1WholebodyHandoverTeleop-v0/dr-level-0 \
    /root/workspace/Haoxuan/RHSCI-runtime/eval-official-pre1102/G1WholebodyOpenFaucetTeleop-v0/dr-level-0 \
  --scene-root /root/workspace/Haoxuan/RHSCI-runtime/external/SIMPLE-official/data/scenes/hssd \
  --output-dir /root/workspace/Haoxuan/RHSCI-runtime/logs/gr00t-old-scene-reference-audit-retry
```

本段覆盖以下所有历史队列。用户要求“先不要no state…先不测新包，测旧包即可…别测pi05”。已TERM旧调度420186，其4个NO-STATE worker/服务/客户端均经原生清理退出；停止旧π0.5父进程1812071/2/3、SIGINT其客户端2195095/2195572/2223472，TERM专用π0.5服务1801072/2156408，现场ps确认这些评测与服务均无残留。所有结果与视频保留，不续跑π0.5，不把中断episode算模型失败。原400次结果另存，不重跑。

当前四组：GR00T烤箱STATE/VGQ-AE、交接STATE、水龙头STATE，均已在原preparation/READY.json校验完成，checkpoint-20000，不再下载权重。官方旧包固定USC-PSI-Lab/psi-data@e78fb93cc28912a3031a10b8656d32d7f0a2b867，仅三个等级，每级10个初态×5轮，每组150次，共600次；hard记NA。新下载仅三个小型旧初态ZIP，共1,176,982 bytes，HF LFS SHA256验证及每任务三级各10条通过。旧初态、场景配置与资源引用来自完整旧包，不仅换USD。运行仿真源码仍是已恢复官方b75，不能称整个历史软件栈复现。准备命令已实际执行：

```bash
cd /root/workspace/RHSCI-Private
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  /root/miniforge/bin/python3 -u experiments/simple_queue/prepare_old_evaluation.py \
  --output-root /root/workspace/Haoxuan/RHSCI-runtime/eval-official-pre1102
```

配置来源先核[SIMPLE论文附录A.3](https://arxiv.org/html/2606.08278v2#A3)：GR00T顺序执行、无RTC；再核[固定官方README](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/baselines/gr00t-n1.6/README.md)与[_common_simple.yaml](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/baselines/gr00t-n1.6/presets/eval/_common_simple.yaml)。官方通用例子max_episode_steps=360不是这些任务的专用时限；本次采用固定b75任务metadata：烤箱1000/0.5，交接800/构造器默认0.9，水龙头1000/0.2。具体16步来自现有RHSCI README、各权重launch_contract.valid_action_horizon=16与原生服务截断；不是论文明确写16。保存config内部action_horizon=50、num_inference_timesteps=4、BF16：模型内部生成50长度张量，其中训练有效前16步，服务返回16、执行16，无RTC。保留原生processor/归一化，不套用π0.5的30步或10次去噪。

首波烤箱STATE占GPU0/1/2，对应level0/1/2、8120/1/2与DDS100/1/2；VGQ-AE占GPU3/4/5，对应8123/4/5与DDS103/4/5。第二波交接STATE、第三波水龙头STATE各GPU0/1/2；不并发复用DDS/端口。每实例8线程，前两轮每个episode双目视频全保留，之后三轮只保留统计；各worker必须真实预热返回有限(16,36)才能启动Isaac。原600次调度输出根：/root/workspace/Haoxuan/RHSCI-runtime/evals/gr00t-official-pre1102-5cycles-20261104。

实际调度入口（plan.py明确四模型、三级旧包和五轮；每worker完整展开命令保存command-gpu*.sh，每轮原生客户端命令保存command.sh）：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh gr00t
cd /root/workspace/RHSCI-Private
python -u experiments/simple_queue/run.py \
  > "$PSI_RUNTIME_ROOT/logs/gr00t-old-package-queue-usd-compat.log" 2>&1
```

调度、worker与模型服务均使用.venv-gr00t-rsci，仿真使用.venv-simple。复用的experiments/pi05_between目录内worker已经支持多模型，不加载π0.5。首波STATE level0实际展开的worker为：

```bash
/root/workspace/Haoxuan/RHSCI-runtime/.venv-gr00t-rsci/bin/python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/run_evaluation.py \
  --gpu 0 --backbone gr00t --variant STATE \
  --task G1WholebodyOpenOvenTeleop-v0 \
  --run-dir /root/workspace/Haoxuan/RHSCI-runtime/checkpoints/followup/gr00t-STATE-oven/inference \
  --checkpoint-step 20000 --expected-horizon 16 \
  --max-episode-steps 1000 --success-criteria 0.5 \
  --evaluation-root /root/workspace/Haoxuan/RHSCI-runtime/eval-official-pre1102 \
  --output-root /root/workspace/Haoxuan/RHSCI-runtime/evals/gr00t-official-pre1102-5cycles-20261104/gr00t-STATE-oven \
  --level dr-level-0 --num-repeats 5 --video-rounds 2 \
  --repair-scene-references
```

其服务是下方历史示例相同GR00T命令。第一轮SIMPLE原生格式客户端，其他任务按本段参数替换：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
cd "$PSI_RUNTIME_ROOT/evals/gr00t-official-pre1102-5cycles-20261104/gr00t-STATE-oven/STATE/dr-level-0/repeat-01"
CUDA_VISIBLE_DEVICES=0 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyOpenOvenTeleop-v0 psi0_decoupled_wbc train \
  --data-format lerobot --sim-mode mujoco_isaac \
  --data-dir "$PSI_RUNTIME_ROOT/eval-official-pre1102/G1WholebodyOpenOvenTeleop-v0/dr-level-0" \
  --host 127.0.0.1 --port 8120 --headless \
  --num-episodes 10 --episode-start 0 \
  --max-episode-steps 1000 --success-criteria 0.5 \
  --eval-dir "$PWD/stats" --save-video \
  --evaluation-round 1 --video-rounds 2 \
  --renderer-gpu 0 --dds-domain-id 100 --cpu-threads 8 \
  --repair-scene-references
```

最新范围覆盖下方历史计划：只续传 GR00T 烤箱 STATE、VGQ-AE；其他指定模型仅使用已完整下载的归档，即 π0.5 NO-STATE 两桌、GR00T 交接 STATE、水龙头 STATE。当前后续共 5 组、1000 次；交接/水龙头 VGQ-AE 暂缓，保留分片。其他本地归档缺失/损坏时直接报错，不触发补下载。全部所选模型完成 SHA256 校验、解包且当前 400 次完成后才启动。

每个模型/任务/难度保留前两轮全部 episode 的左右目视频，不按成功率筛选。worker 默认 `--video-rounds 2`，后面三轮关闭录像，但继续保存统计/日志/命令。结果用 level-0 / level-1 / level-2 / hard 的成功数和已完成数向量汇报，读取命令见两桌 README。现有等待进程启动未来 worker 时会读取新的 worker/客户端文件，无需重启当前评测或下载。

最新实际准备/等待命令（分别后台运行；日志 `logs/followup-preparation-selected.log`、`logs/followup-queue-selected.log`）：

```bash
cd /root/workspace/RHSCI-Private
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  /root/miniforge/bin/python3 -u experiments/simple_queue/prepare.py
/root/workspace/Haoxuan/RHSCI-runtime/.venv-pi05-rsci/bin/python \
  -u experiments/simple_queue/run.py
```

原准备/等待进程已停止，保留烤箱原下载进程续传，未停止当前八路评测。下方原始七组计划和旧 PID/日志用于溯源，以本段范围和工作记录为准。模型配置、官方来源与实际 SIMPLE 命令不变。

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
