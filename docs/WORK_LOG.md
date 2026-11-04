# 必要工作记录

每次开始工作前必须完整阅读本文件、`AGENTS.md` 及 `ERROR_LOG.md`。这里维护后续工作所需的当前事实；实验具体参数和完整命令维护在对应已登记文档，运行日志和大产物放在仓库外。所有状态区分准备、启动、实际 rollout 和完成。

## 当前仓库与环境

- 工作仓库 `/root/workspace/RHSCI-Private`，分支 `main`；仅本地提交，不自动推送。模型使用现有 `src/<模型>/rhsci` 和公共 `src/rhsci`，实验编排放 `experiments/<主题>/`。
- 运行目录 `/root/workspace/Haoxuan/RHSCI-runtime`，`/opt/rsci-data` 指向此处。`source <运行目录>/activate.sh pi05` 选择模型环境，`source <运行目录>/activate.sh simple` 选择仿真环境；激活脚本会改变工作目录。
- 8 张 RTX 5090，每卡约 32 GiB。SIMPLE 是官方 `b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11`，9 个递归子模块固定在 gitlink；官方跟踪源码未修改。Isaac Sim 4.5.0.0、MuJoCo 3.3.6。59 个官方场景/机器人/材质归档已恢复。
- 模型环境 `.venv-pi05-rsci`：Python 3.10.12、Torch 2.7.0+cu128、Transformers 4.53.2，已安装仓库 overlay。8 个模型均通过原生严格权重加载及真实 `(30, 36)` 推理预热，无需更换编译 backend。

## 当前 π0.5 两桌任务

- 旧包完整对照最终使用独立VGQ-AE服务GPU7/8147，三个仿真GPU4/5/6、DDS210–212，产物 `evals/pi05-vgq-official-pre1102-dedicated-20261104`。首次共享服务启动已技术中止并保留日志，避免原worker提前关闭模型；不混入成功率，不影响原八路。独立服务加载及三路rollout状态须以实际日志核实。

- 用户最新明确旧场景指官方11月2日同步前、仅三个等级的完整评测包，且VGQ-AE历史约30%（历史权重/命令未核实）。已下载官方e78fb93旧包SHA256=8a4b0ff7…90e7e0，解包75文件与此前镜像旧包全部相同；不能以单条120步探针代替完整旧包成功率。已准备独立完整复测脚本run_old_eval.sh：三等级各10初态×5轮=150次，30/30无RTC、10步BF16、1200步/0.4，前两轮双目录像，复用VGQ8124–8126，GPU4–6/DDS210–212；命令与来源见两桌README。固定噪声10/20/50步积分对照已完成，step30首目标仍约-0.03而实际-0.119，不是已验证修复。

- 用户明确要求继续找根因，并更正“旧评测数据”指旧 task-layouts 的初态/布局等；上轮模型输出回跳只是直接机制，不能当根因已查清。持续 goal 已重建。旧初态已实际完成 VGQ-AE 120 步闭环，仍回跳（边界/内部比 13.08）；灯光恢复训练值也未消除（20.74）。训练完整初态为8.79；前移出生点会在策略前碰落盒子，官方PR29已记载约-6.1cm为WBC正常settle。全程仿真时钟也只延后尖峰。专家动作回放已完成且换块无额外尖峰（比值0.97），30/60/90步真实双臂相对训练记录RMSE为0.00083/0.00276/0.00624rad。同图交换状态证明STATE可随当前关节保持进度，VGQ-AE当前图像却仍退向早期姿态；故障已收窄到本权重的视觉闭环反馈不足，尚未确认具体训练成因或可修改的部署代码bug。GPU1固定噪声10/20/50步采样数值对照进行中，在线服务未动。配置、来源与完整命令见必读两桌README，产物 `evals/stutter-root-cause-20261104`。正式八路模型/初态/材质/时钟均未改，临时专家服务8140已关闭。

- 用户报告动作卡顿，已启动 goal 并完成三个独立 120 步真实诊断：同一 level-2/episode-3，原 VGQ-AE、STATE 输入对照、VGQ-AE 输入对照；只复用现有服务，DDS 196/197/198，均已结束，未中断正式八路。配置仍预测/执行 30/30、无 RTC、10 步采样、50 Hz；实际完整命令与来源沿用必读实验说明。产物根 `evals/stutter-diagnostic-20261104`，含逐步 trace、输入图、视频、数值 JSON 和轨迹 PNG；截短诊断不计正式成功率。
- 卡顿直接机制已定位到模型动作块边界目标回跳：两次 VGQ-AE 边界跳变约块内变化的 21.25/23.04 倍，STATE 对照约 4.12 倍；原始模型动作到 WBC 双臂输入逐步最大误差均为 0。输入观测与 MuJoCo 双臂角最大差小于 1.2e-6 rad，图像随姿态更新；实际机械臂跟随回跳发生反向。不能仅凭这一初态前 120 步判定所有场景、权重和训练均无问题；具体学习/兼容原因未确证，不宣称已经修复。
- 本轮只加实验性 opt-in 记录与分析，不改正式模型、场景、控制时钟或 30/30；也不加 RTC/平滑。用户指定原版场景资源对照为找不到其他原因时的最后手段，尚未执行。原生插值器重放显示改用仿真时钟只是尖峰延后一帧，不是有效修复；详见错误记录。
- GPU 1 独立编译数值对照已完成并退出（原 PID 508834，正式服务未改）：固定 seed=17 对同一真实输入比较 eager / max-autotune，四张图的双臂最大单关节差 0.004512 rad、最大联合 L2 差 0.009599 rad；并非逐位相同，但不足以解释 0.21–0.29 rad 的边界回跳。第 30 步实际左肩 -0.11909 rad，eager 下一块起点 -0.03072 rad，也会明显反向。完整命令见两桌 README，结果 `evals/stutter-diagnostic-20261104/inference-mode-comparison.json`；首次 meta 初始化错误及失败日志保留，后续实际对照正常结束，不把候选 cc=120 报错单独作为整次推理失败。
- 原场景资源只读对比：RHSCI `assets-20261016/assets-manifest.json` 中本次场景 `102344022` 加全部 vMaterials 文件共 6014 项，与当前场景/材质的文件名集合和逐文件 SHA256 完全一致，缺少/多出/内容不同均为 0；结果 `original-scene-comparison.json`。没有替换资源或额外做资源 A/B，不将该范围外的物体/机器人资产声称为本次已比对。对已观察到的本样本卡顿，直接机制定位为模型块接续目标回跳，未确认需要修改的实现错误，不通过改变评测策略消除现象。
- 本轮诊断收尾正式阶段统计（level-0/1/2/hard）：STATE 成功 `0/0/0/0`、已完成 `7/8/7/8`；VGQ-AE 成功 `0/1/1/0`、已完成 `8/8/9/7`。仍未跑满每模型 200 次。opt-in trace 与分析器经三份真实轨迹检查，额外编译对照实际运行完成；四个 Python 文件 Ruff lint/format 通过，多 episode trace 在 Kit 导入前拒绝以免混淆步号。

- 最新视频/统计约定已写入 AGENTS 与评测入口：每模型、每难度保留前两轮全部 episode 左右目视频；之后不录像，不删除已有文件。当前旧 worker 无需重启，后续每轮客户端依据 repeat-NN 目录落实规则；新 worker 显式记录轮次和 video_rounds=2。结果按 level-0/1/2/hard 分别汇报成功数与已完成数。
- 本次复查阶段快照两模型均完成 `4/4/4/4`；STATE 成功 `0/0/0/0`，VGQ-AE 成功 `0/0/1/0`。已有 64 份带 success/failed 后缀的左右目 MP4；成功视频为 VGQ-AE/level-2/episode-3，640×360、50fps、1077 帧、21.54s。成功/失败各一份完整 FFmpeg 解码通过，成功视频 4 帧拼图 `logs/pi05-success-video-review.png` 已人工检查。实际日志未见新 USD/纹理错误、traceback 或 OOM，仍有官方篮子空碰撞 mesh 告警与 torchvision 弃用警告。核对了 30 步响应、队列耗尽后重预测、reset 清队列及原生成功判定，不将低成功率直接判作程序 bug。
- 后续逐份检查已关闭的 66 个 MP4，ffprobe 均返回有效 640×360、帧数和时长，与对应官方结果及左右目标签一致；此时 VGQ-AE 已完成 `4/4/5/4`、成功仍 `0/0/1/0`。录像规则 1–5 轮与显式轮次覆盖、重复/越界/损坏统计拒绝均经 CPU 检查。等待队列存在“worker 意外死亡但缺完成标记时一直等待”的静态/CPU 可复现缺口，未改动故障处理，详见错误记录；当前八路仍在运行。
- 收尾快照：STATE 成功 `0/0/0/0`、已完成 `4/5/4/5`；VGQ-AE 成功 `0/0/1/0`、已完成 `4/4/5/4`，均为第一轮阶段结果。四个修改/新增 Python 文件通过 Ruff lint/format；使用 mock 官方接口验证旧 worker 的 repeat-03 客户端实际关闭录像并在导入 Kit 前记录有效配置，没有额外启动 GPU 评测。

完整配置、来源和命令见 [实验说明](../experiments/pi05_between/README.md)，已在 `AGENTS.md` 登记为必读。

- 用户指定 STATE 与 VGQ-AE，各 checkpoint-40000，预测 30 / 执行 30，无 RTC；四难度各 10 个初态重复 5 轮，每模型 200 次，总计 400 次。重复不是 200 个独立初态。
- 模型 HF revision `59a1ea14d4346496d4e55c9fa58ea2f3330fd298`；上述两份推理权重已下载并 SHA256 验证。用户最新要求只下载已指定评测权重，其余全量下载调度、自动重试器和无关传输均已停止，保留已下载内容和分片，不自动恢复。
- 用户指定 dataset revision `ea9d480ff1d95f34b63d7ea368b1bc72e353fd57`；eval-data 共 151 文件、8,045,136,314 字节已全部验证。实际初态用最新官方 `SIMPLE-eval/fb67cd9a309cb4f3ce8f9470fecb6fa6d82c60d4`，2993 文件已验证，与 RHSCI 镜像一致。
- 匹配的两桌训练数据已下载、校验并解包，用于原处理器和统计；实际 fingerprint `b16d2e98348482834e282f2aa39ecc9154a60a6d6da4dad703f9ba9cc6141e2e` 与两份权重契约均匹配。不能用新版训练统计替换它。tokenizer SHA256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` 已验证。
- 已执行启动入口 `bash experiments/pi05_between/launch_8gpu.sh`。GPU 0–3 为 STATE 四难度，4–7 为 VGQ-AE 四难度；端口 8120–8127，DDS domain 100–107。首次尝试因 USD 诊断发现失效引用而主动停止；输出完整保留于同名 `-before-usd-repair` 目录，不计入正式 400 次。四难度渲染通过后已按相同命令重启，尚不能据此宣称完成物理评测。
- 输出根 `<运行目录>/evals/pi05-between-30x30-5cycles-20261104`。每卡有 `worker.json`、`server.log`；每轮有实际 `command.sh`、`console.log`、统计和视频。每卡完整 50 次后才产生 `COMPLETE.json`。
- 本机 keepalive 用 `/root/bin/xiancun pause 360` 暂停并释放 GPU；不要直接杀其 worker，监督器会重启。未停止其他生产训练。

## 已授权的后续评测队列

- 当前新准备 PID 419978、等待队列 PID 420186；日志分别为 `logs/followup-preparation-selected.log`、`logs/followup-queue-selected.log`。实际检查仅烤箱两个下载进程存活，准备进程等待原续传、并行核验本地水龙头 STATE；队列等待当前 400 次和所选资产。代码语法、5 组选择范围、只允许 2 份下载及 worker horizon 参数检查通过。

- **最新范围覆盖本节旧计划：** 只继续下载 GR00T 烤箱 STATE / VGQ-AE，原下载 PID 267262 / 267263 保持续传。已停止旧准备 PID 267176、旧等待 PID 288997 和交接 VGQ-AE 下载 PID 376533，保留所有文件/分片，当前八路评测未停。其他指定模型只评测已下载完整且本地 SHA256 通过的权重：π0.5 NO-STATE、GR00T 交接 STATE、水龙头 STATE。后续共 5 组、1000 次，交接/水龙头 VGQ-AE 暂缓；准备与队列采用 `*-selected.log`，实际命令见队列 README 最新段落。非烤箱归档缺失/损坏不自动补下载。
- 本次检查官方 `eval_stats.txt` 的阶段快照：STATE 四难度依次完成 3/4/3/4 次，均无成功，共 0/14；VGQ-AE 完成 4/3/4/3 次，level-2 成功 1 次，共 1/14。全部仍在 repeat-01，没有 `verified_results.json` 或完整模型结果。阶段样本不能代表最终 200 次成功率。

- 用户追加 7 组：π0.5 NO-STATE 两桌，及 GR00T 开烤箱/交接/开水龙头各 STATE、VGQ-AE；每组仍 5 轮、200 次，共新增 1400 次。先等当前 8 路均完成 50 次且逐轮核验结果，再按任务接续。
- 配置与来源、完整命令见已登记的 `experiments/simple_queue/README.md`：π0.5 30/30、40k；GR00T 16/16、20k，全部无 RTC。各任务使用自己的官方时限与成功判定，不套两桌参数。
- CPU 准备进程 `experiments/simple_queue/prepare.py` PID 267176，日志 `<运行目录>/logs/followup-preparation.log`。π0.5 NO-STATE 已完成下载/SHA256/解包及 horizon=30 核对，收据 `evals/followup-pi05-gr00t-5cycles-20261104/preparation/pi05-NO-STATE-between.json`。GR00T 交接 STATE 也已完整准备、horizon=16；水龙头 STATE 归档已下载/SHA256 通过。其余 4 份尚未全部就绪。用户限定只下载指定模型；全量调度 PID 86344、重试器 96854 及四个无关传输已结束，源码已取消自动恢复。
- **当前下载已恢复：** 用户最新明确专用节点不可用就换回、以能下载为优先。已将下载组切回香港 Y01，真实烤箱 STATE 权重 Range 请求返回 206、`bytes 0-65535/17701737683`、65536 字节，约 1.35 秒；再核对 PID 身份后 SIGCONT 准备进程 267176 和两个烤箱下载进程 267262、267263，复用分片/解包现场，无重复下载任务。GR00T 交接 STATE 已完成解包及 horizon=16 核对。只继续指定权重，当前评测及等待队列始终未暂停。
- 自动等待队列 `experiments/simple_queue/run.py` 已启动，日志 `<运行目录>/logs/followup-queue.log`；它等待当前真实 400 次及全部后续权重就绪。不会以进程退出码/health/预热代替 episode 完成。
- 队列每小时调用本机 `xiancun pause 360` 续暂停空闲占卡程序，防止长评测中六小时暂停到期；队列结束停止续期，原监督服务不删除、不禁用。
- GR00T 专用环境 Torch 2.7.0+cu128、Transformers 4.51.3、FlashAttention 2.7.4.post1 导入通过；后续真实 GPU 预热尚未执行，不能称已验证闭环。

## 材质检查当前状态

- 单卡双路独立容量探针已结束：GPU 2 / slot 1，端口 8130 / DDS 110，第二路模型预热通过；场景加载时总占用达到 30336 MiB、余量 1774 MiB，触发预设 2048 MiB 保护，仅停止额外 worker，未发生已观测到的 OOM。实际 episode 未完成，不计正式 400 次，也不能称双路稳定。结果 `evals/dual-gpu2-capacity-probe-20261104/result.json`；没有据此全卡加倍。原 8 路继续运行。
- 官方文件清单复核：50 ZIP、34068 文件缺少/多出/大小或 CRC 不符均为 0；88 处别名与官方 props/textures 内容一致。结果 `logs/material-inventory-audit.json`。**全部 40 初态实际 reset/render 已通过**：USD composition errors=0，80 处 table/table2 检查的 MDL 路径/导出名与官方 layout 一致；官方未绑定材质的 table2 仍未绑定。结果 `logs/pi05-between-material-rendering-all40/result.json`；日志 `logs/material-all40.log` 无缺失 USD/纹理/UsdToMdl/traceback。原生 close 已正常退出，诊断不计正式评测次数。

静态检查覆盖 40 初态、80 处材质使用、63 个 MDL 文件与导出名称、175 个纹理路径，未发现缺失。实际 USD 渲染另复现了 `/root/tmp` 失效旧软链接及缺失纹理，已检查/恢复 88 处 props/textures 别名。后续全 40 初态真实渲染和桌面绑定已通过，详见上列新收据；四初态旧收据保留溯源。后续正式 rollout 日志仍须检查；篮子碰撞 mesh 无 points 属官方资产已知限制，不归为材质问题，也未声称该物理资产无错误。

## 下载代理与流量约定

- Mihomo 配置 `/root/workspace/Haoxuan/mihomo/config/config.yaml` 已原生 `-t` 校验并通过本地 API 热加载；备份 `.before-download-route-20261104`。不要将代理密码、UUID、secret 或订阅链接复制进仓库/日志。
- “📥 下载专用” selector 现含香港 Y01、日本 Y05/Y06；**当前和重启默认均为香港 Y01**，按用户新授权优先保证可下载。HF `huggingface.co`/`hf.co`、PyPI、Torch、NVIDIA 下载域和 GitHub release/codeload 域继续走该组；普通“🔰 选择节点”仍为香港 Y01。x0.01 仅是日本节点标注，不代表已查实际账单。低倍率节点日后实际恢复再考虑切换；更新订阅可能覆盖本地规则，更新后重查。
- 复查节点：Y05 可解析但目标端口 TCP 超时；Y06 本机 DNS 返回 -2，不过 Google 与 Cloudflare DoH 均返回 NOERROR/3 条答案，解析得到的 A 地址直连其端口仍超时，所以不只是本机 DNS。两节点 HF 延迟测试分别返回 504/503，尚不能区分服务端离线或中间线路屏蔽，不猜改凭据或协议。香港 Y01 TCP 成功、HF 延迟 311ms、真实权重 Range 通过，已据此恢复下载。
- 恢复后连接审计已确认 `us.aws.cdn.hf.co` chains 为 `香港Y01 → 📥 下载专用`，且连接下载字节和烤箱权重分片数继续增加，不仅是进程处于运行状态。
- 已断开 23 条旧 HF 代理连接，避免切节点后旧 TCP 会话继续走香港 Y01。已下载分片全部保留。恢复前先用小 HEAD/Range 验证实际 HF CDN 路由，再核对下载进程身份、续传并检查 `/connections` 的 chains。
