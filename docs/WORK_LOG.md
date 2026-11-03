# 必要工作记录

每次开始工作前必须完整阅读本文件、`AGENTS.md` 及 `ERROR_LOG.md`。这里维护后续工作所需的当前事实；实验具体参数和完整命令维护在对应已登记文档，运行日志和大产物放在仓库外。所有状态区分准备、启动、实际 rollout 和完成。

## 当前仓库与环境

- 工作仓库 `/root/workspace/RHSCI-Private`，分支 `main`；仅本地提交，不自动推送。模型使用现有 `src/<模型>/rhsci` 和公共 `src/rhsci`，实验编排放 `experiments/<主题>/`。
- 运行目录 `/root/workspace/Haoxuan/RHSCI-runtime`，`/opt/rsci-data` 指向此处。`source <运行目录>/activate.sh pi05` 选择模型环境，`source <运行目录>/activate.sh simple` 选择仿真环境；激活脚本会改变工作目录。
- 8 张 RTX 5090，每卡约 32 GiB。SIMPLE 是官方 `b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11`，9 个递归子模块固定在 gitlink；官方跟踪源码未修改。Isaac Sim 4.5.0.0、MuJoCo 3.3.6。59 个官方场景/机器人/材质归档已恢复。
- 模型环境 `.venv-pi05-rsci`：Python 3.10.12、Torch 2.7.0+cu128、Transformers 4.53.2，已安装仓库 overlay。8 个模型均通过原生严格权重加载及真实 `(30, 36)` 推理预热，无需更换编译 backend。

## 当前 π0.5 两桌任务

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
