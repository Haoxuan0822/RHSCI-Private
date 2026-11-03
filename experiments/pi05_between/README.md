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

用户反馈动作卡顿后追加诊断：复用 GPU 6 现有 VGQ-AE 模型服务（端口 8126），另启独立仿真和 DDS 196，不加载第二个大模型；level-2 episode 3，预测/执行 30/30、无 RTC、10 步采样均不变。仅截取 120 控制步以记录前 4 块，保留视频，属于诊断而非正式成功率；`trace_control.py` 只包装原生方法记录每块动作、WBC 目标/输出、仿真时间和真实时间，不更改控制时钟。实际命令：

后续两个输入对照沿用同一完整命令和 120 步，只替换输出目录、GPU/服务/DDS：`state-observation` = GPU 0、port 8120、DDS 197；`vgq-observation` = GPU 6、port 8126、DDS 198。新增记录每块实际输入状态/图像，以及 MuJoCo 与观测关节角；不更换场景资源。原诊断已完成 120 步，4 块推理约 0.068–0.082 秒，每步仿真 0.02 秒；双臂块间目标跳变 0.212/0.203/0.284 rad 的联合 L2，块内中位 0.010 rad，约 21 倍。CPU 原生插值器重放证明：换仿真时钟只会将边界尖峰后移一帧，不能解决模型目标回跳，不能据单个边界帧变小就声称时钟修复有效。

三个诊断均已完成。输入对照中 STATE 的三次双臂边界跳变为 0.0224/0.0726/0.0404 rad，块内中位 0.00982 rad，边界中位约 4.12 倍；VGQ-AE 为 0.2364/0.2880/0.2086 rad，块内中位 0.01026 rad，约 23.04 倍。三份 trace 每个已执行动作的原始模型双臂值与 WBC 输入逐项完全相同，最大误差为 0，排除本样本的双臂队列错位或转换引入回跳。观测与 MuJoCo 双臂最大角度误差分别为 1.17e-6 / 5.66e-7 rad；实际输入图像已保存并人工查看，图像会随姿态更新。VGQ-AE 左肩 yaw 在第 29 步目标约 -0.116 rad，下一块起点约 -0.022 rad，真实关节随后也反向。这定位了已复现卡顿的直接机制：30 步块接续时模型目标不连续；尚不能据此断言权重、训练或所有输入兼容细节均无问题，也没有将其归因为已证实的场景资源错误。

两个输入对照的初始状态逐项相同，但独立仿真渲染的初始图像并非逐像素相同，不能称严格同输入模型配对。VGQ-AE 相邻请求图像像素平均绝对差为 7.55/2.45/1.40（0–255），STATE 为 7.74/11.32/10.03；这是图像有更新的证据，不替代相机全链路时序验收，收据 `input-check.json`。归档 STATE 源码的 `encode/expert/sample_actions` AST 与当前一致，收据 `inference-source-comparison.json`；该归档不是经过身份核验的 VGQ 源码，也不替代全模型数值对照。

未改变模型、30/30、RTC、正式评测、控制时钟或场景材质；没有加平滑掩盖回跳。根据用户最新顺序，原版场景资源替换仍作为最后手段，当前未执行。截短诊断的视频 failed 标签不计正式失败或成功率。仅解码/帧数正确不足以验证动作连续性，此前视频格式复查不能替代本次控制轨迹检查。

数值与图在运行根 `evals/stutter-diagnostic-20261104/{original,state-observation,vgq-observation}/control_analysis.{json,png}`，输入拼图为同级 `input-comparison.png`。重算命令（simple 环境提供 NumPy/Matplotlib，无需安装新依赖）：

```bash
cd /root/workspace/RHSCI-Private
/root/workspace/Haoxuan/RHSCI-runtime/.venv-simple/bin/python \
  experiments/pi05_between/analyze_control_trace.py \
  --trace /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-diagnostic-20261104/vgq-observation/control_trace.jsonl \
  --plot
```

原始 120 步诊断的完整启动命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
mkdir -p "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/original"
cd "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/original"
CUDA_VISIBLE_DEVICES=6 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir "$SIMPLE_EVAL_ROOT/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-2" \
  --sim-mode mujoco_isaac --host 127.0.0.1 --port 8126 --headless \
  --num-episodes 1 --episode-start 3 \
  --max-episode-steps 120 --success-criteria 0.4 \
  --eval-dir "$PWD/stats" --save-video \
  --renderer-gpu 6 --dds-domain-id 196 --cpu-threads 8 \
  --trace-control "$PWD/control_trace.jsonl"
```

视频保留规则：每个模型、每个难度保留 repeat-01 / repeat-02 的全部 10 个 episode，左右目均保存（每模型两轮共 160 个 MP4）。repeat-03–05 不录像，统计/日志/命令照常保留，不删除已有视频。`run_evaluation.py --video-rounds 2` 默认落实该规则，客户端显式接收 `--evaluation-round N --video-rounds 2`；为兼容仍在运行的旧 worker，客户端也可从 `repeat-NN` 工作目录推断轮次，无需中断当前评测。客户端在 Kit 导入前保存 `effective_eval_config.json` 和 `VIDEO_POLICY`，实际 `save_video` 以此为准。当前运行中的第一轮已按旧入口开启录像，之后新启动的每轮客户端读取新规则。

追加独立编译数值对照：GPU 1，复用 VGQ-AE 权重和四张真实输入，固定噪声 seed=17，BF16、10 步采样、30 步输出，对比 eager 与 max-autotune；不接控制器，不计 episode，不更改在线服务。来源仍为本节已核对的保存配置与原 README；指令显式使用匹配训练数据 `meta/tasks.jsonl` 的 task-0 文本，不声称该独立对照重放了在线服务的原随机噪声。12 GiB 可用显存预检不通过则拒绝启动。实际命令：

对照已正常结束。四张输入上双臂最大单关节误差 0.004512 rad、最大联合 L2 误差 0.009599 rad；有 BF16 执行差异，不是逐位相同。eager 第 30 步左肩下一块起点为 -0.03072 rad，此时实际观测为 -0.11909 rad，关闭编译后仍有明显反向目标。结果 `inference-mode-comparison.json`；不能把候选内核 cc=120 错误当成这次推理失败，完整输出已返回。额外进程已退出，显存已释放。

同时只读核对 RHSCI 原资源 manifest：`hf/RHSCI/dataset-ea9d480f/eval-data/SIMPLE/assets-20261016/assets-manifest.json` 中当前 episode 场景 `102344022` 与全部 `vMaterials_2` 的 6014 个文件，与现用资源文件名集合及 SHA256 完全一致，缺少/多出/内容不同均为 0；收据 `original-scene-comparison.json`。这限定于该场景和材质，不替代原物体/机器人资产的比较。本轮没有更换资源或声称完成资源替换 A/B，也没有修改模型/RTC/控制时钟/30 步执行配置；诊断定位直接机制，并非已经修复某个已确认的代码错误。

独立编译数值对照的完整启动命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
TORCHINDUCTOR_COMPILE_THREADS=4 \
TORCHINDUCTOR_CACHE_DIR="$PSI_RUNTIME_ROOT/cache/stutter-compile-check" \
python -u experiments/pi05_between/compare_inference_modes.py \
  --checkpoint "$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1/checkpoint-40000" \
  --trace "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/vgq-observation/control_trace.jsonl" \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --instruction " pick up the cracker box from table1,locomotion to table2,and place  on table2." \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/inference-mode-comparison.json"
```

实际录像路径为 `模型/难度/repeat-NN/data/evals/psi0_decoupled_wbc/任务/难度/episode_N/head_stereo_{left,right}_{success,failed}.mp4`。没有 success/failed 后缀的文件可能仍在写入，不能把未关闭的视频算作完整产物。

结果始终按 level-0 / level-1 / level-2 / hard 排序，分别报告成功数和已完成数向量；例如成功 `0/0/1/0`、已完成 `4/4/4/4`，不能将未运行样本算失败。使用只读命令：

```bash
python3 experiments/pi05_between/summarize_results.py \
  --output-root /root/workspace/Haoxuan/RHSCI-runtime/evals/pi05-between-30x30-5cycles-20261104
```

本次复查已核对官方动作队列为空才请求下一块、upsample_factor=1、每 episode reset 清队列与历史标志，当前日志持续收到 30 步动作；统计与已关闭视频 success/failed 后缀一致。成功/失败样本各一份完整 FFmpeg 解码通过，成功视频抽帧已人工检查。未发现新致命异常或材质引用错误，不据此宣称不存在任何 bug；官方篮子空碰撞网格和手指映射限制仍保留。

固定模型 revision `59a1ea14d4346496d4e55c9fa58ea2f3330fd298`，训练数据来源为用户指定 dataset revision `ea9d480ff1d95f34b63d7ea368b1bc72e353fd57`。下载完成核对 HF SHA256，再恢复模型配置、训练统计和 tokenizer；模型不需要 optimizer/RNG 即可推理。

运行根为 `/root/workspace/Haoxuan/RHSCI-runtime/evals/pi05-between-30x30-5cycles-20261104`。`worker-gpu*.log` 为编排日志，每卡目录有 `server.log`，每轮有 `console.log`、`stats/` 和视频。Kit 关闭可能提前退出 Python，因此每轮客户端退出后额外核验官方 `stats/eval_stats.txt` 中 episode 0–9 恰好各一条，再生成 `verified_results.json`；不以关闭之后可能未写出的 `result.json` 或退出码 0 代替。仅有 `warmup.json` 不能视为物理评测完成；每卡五轮均验证共 50 次后才写 `COMPLETE.json`。

## 材质与 USD 引用检查

2026-11-04 追加按官方 ZIP 原始清单复核：49 个场景包与 1 个材质包，共 34068 个文件，文件名集合缺少 0、多出 0、大小/ZIP CRC32 不符 0；88 处 props/textures 别名与其对应官方目录文件集合和内容一致。归档 SHA256 在恢复时已验证，本次文件内容比较为大小与 CRC32，不将其称为再次全量 SHA256。官方材质包本身含有嵌套目录，按原样保留。命令为 `python experiments/pi05_between/audit_material_inventory.py`，完整结果在运行根 `logs/material-inventory-audit.json`。

全 40 初态的实际 reset/render 与桌面 MDL 绑定检查现使用下列命令，结果逐初态写入 `logs/pi05-between-material-rendering-all40/progress.json`，全部通过后才生成 `result.json`；这是资源诊断，不计入正式评测。检查有效绑定的 MDL 路径和导出名称，不能把“MDL 文件存在”当作绑定正确。官方保存的 `ground_material` 字段在当前 DR 实现中没有应用，保持官方行为，没有自行额外绑定地面材质。

完整结果已通过：四难度各 10 个初态共 40 个真实渲染、80 处桌面/第二桌面绑定核对，composition errors 为 0，材质 MDL 路径/导出名称与官方 layout 一致，未绑定的第二桌面未多绑定；日志无缺失 USD/纹理/UsdToMdl/traceback。收据为上述目录的 `result.json`，诊断进程已正常退出。篮子碰撞 mesh 错误仍按官方已知限制保留，不将其误报为材质错误。

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
