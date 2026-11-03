# 错误记录

每次开始工作前必须完整阅读本文件和 `AGENTS.md`。错误按实际证据记录，不用静态检查、下载成功或进程存活替代实际运行验证；修复后更新原条目中的状态与验证范围。

## USD 引用旧路径失效：已修复，全部 40 初态真实渲染与桌面绑定通过

- 现象：实际 Isaac 渲染日志出现 `Could not open asset @/root/tmp/bdb1f316aa0188ec/props/TV_Screen2.usd@`，同时还有 Shader11、MIRROR 等 USD 子资源引用错误。不能把它归为不影响场景的普通启动噪声。
- 原因：官方场景 USD 内含 `/root/tmp/<编号>/props` 等旧绝对路径；本机该路径仍是指向已删除旧运行目录的失效软链接。官方加载器准备的 `/tmp/<编号>/props` 及当前官方源资源实际存在，但不会自动修正 `/root/tmp` 下的旧链接。
- 遗漏：此前修复只扫描 `/tmp`，没有覆盖 `/root/tmp`；默认日志级别还会省略部分警告。材质文件存在、烟雾测试能出图，不代表 USD 组合完整。
- 处理：`experiments/pi05_between/repair_usd_paths.py` 扫描官方 HSSD USD 中实际引用的 `/tmp`、`/root/tmp` 路径；仅将失效链接改名保留，再建立指向当前官方源资源的兼容链接，不改官方 USD 文件。正式评测客户端开启 warning 级别保留引用警告。
- 验证：88 处别名均存在；四难度各 episode 0 完成真实 reset/render，逐 prim composition errors 全为 0、图像均为有效 `(360, 640, 3)`。`logs/pi05-between-material-rendering-textures-restored.log` 中 `Could not open asset`、缺失纹理、`UsdToMdl`、`Failed to resolve`、traceback 全为 0；`logs/pi05-between-material-rendering/result.json` 保存结果，level-2 图像已人工查看。范围是四个初态的实际渲染与全部 40 初态的静态 MDL 检查，不宣称所有后续 rollout 已完成；后续仍检查正式日志。
- 防止复发：扫描资产中真实嵌入的所有绝对路径，而不是只猜一个临时根；同时检查 MDL 文件、导出名称、纹理、USD 子引用与实际图像。保留首次失败日志及旧链接，明确验证覆盖哪些初态。
- 补充发现：USD crate 的引用可能被压缩，`strings` 返回零条不能证明没有引用；实际相对路径可解析至 `/root/tmp`。修复脚本已改用 `Sdf.Layer.GetExternalReferences()`，解析并恢复了 44 处场景引用别名。渲染诊断使用了 Isaac Sim 4.5 未提供的 `Stage.GetCompositionErrors()`，又因 Kit 的 `close()` 掩盖 traceback，导致初次看不到失败原因。现改为逐 prim 读取 `GetPrimIndex().localErrors`，先保存结果或 traceback，再关闭；不要拿其他 USD 版本的 API 当作本环境事实。
- 为避免错误场景计入评测，已主动停止首次 8 卡启动，并将完整输出保留为 `evals/pi05-between-30x30-5cycles-20261104-before-usd-repair`。这次尝试不计入正式 400 次。
- 分层复验继续发现 `../textures/*.png` 缺失：只恢复 props 会使 USD composition errors 清零，却仍缺背景纹理。修复已改为同时恢复每组 props 与 textures；必须再看 `[UsdToMdl]` 和资产解析日志，不能只用 composition errors 判定材质正常。
- 后续完整复验：`logs/pi05-between-material-rendering-all40/result.json` 含四难度各 10 个初态，40 次真实 reset/render，USD composition errors=0；80 处 table/table2 有效绑定检查均与官方 layout 的 MDL 路径/导出名一致，无材质的 table2 未额外绑定。`logs/material-all40.log` 的缺失 USD/纹理、UsdToMdl、traceback 均为 0，程序已正常关闭；level-1 episode 7 图像已人工检查。官方 ground_material 字段未实际应用，保持官方实现，没有自作主张增加地面随机材质。

## 下载节点与流量：专用节点暂不可用，已按用户授权回香港续传

- 用户要求以后大文件走下载专用节点。发现日本 Y05/Y06 均标注“下载专用 | x0.01”，此前 HF/CDN 的实际连接链却是香港 Y01。已增加独立下载组和优先域名规则，保留普通流量原选择，未把节点标签当作实际套餐计费验证。
- Y05 TCP 超时，Y06 DNS 解析失败；两个节点的 HF 实际 HEAD、HF robots、gstatic 测试全部失败。已暂停准备/下载进程 267176、267262、267263，并仅断开旧 HF 连接，不自动回退普通节点。文件和分片保留；专用线路恢复且实际小请求验证后才能续传，不能将规则配置成功写成下载已恢复。
- 最新复查与用户授权覆盖上段暂停条件：Y05 DNS 成功、TCP 仍超时；Y06 本机解析失败，但 Google/Cloudflare 两个公共 DoH 都能解析，绕过本机 DNS 直连其 A 地址端口仍超时。说明并非 HF/权重故障，也不能把 Y06 简单断言为域名失效；服务端或中间线路具体原因尚未证实。香港 Y01 TCP/HF 测试通过。用户明确“能下载才是重要的”后，已加入香港为下载组首选、原生配置校验/热加载/选中成功；真实烤箱 STATE 权重 64 KiB Range 返回正确 206、长度与 Content-Range，随后核对身份并恢复三个原进程，保留续传，未恢复无关下载。
- 新规则初次与原 YAML 的单空格列表缩进不一致，原生校验失败；两次失败均未加载。改为与既有列表一致后 `mihomo -t` 通过，才用本地 API 热加载。后续修改先保留受限权限备份、确认列表缩进、原生校验，再加载；不打印含认证信息的代理配置。

## RTX 5090 首次 π0.5 编译候选内核不支持：真实推理已通过

- 现象：Torch 2.7 / Triton 3.3 的 max-autotune 预热中，部分候选报 `computeCapability not supported`、`PassManager::run failed`，日志显示 cc=120。
- 已知范围：原生模型严格加载通过；编译调用仍在执行，栈检查确认在图处理/代码生成中，并非仅凭日志推断卡死。候选失败不等于整个推理失败，也不能在实际 `/act` 返回前称兼容验证通过。
- 处理：保留原推理公式与编译设置，完成真实请求验证；没有通过删日志、换采样步数或假模型掩盖错误。
- 后续证据：8 个模型均实际返回有限的 `(30, 36)`，原生首次编译预热通过，无需切换编译 backend。候选内核失败与完整推理失败必须分开判断。

## 其他仍需保留的已知限制

- **动作一卡一卡：直接机制已复现，尚未宣布修复。** 对用户视频反馈不能仅做 FFmpeg 解码、帧数或进程检查。三个同初态 120 步实际诊断显示第 30/60/90 步新块的模型双臂目标回跳；两次 VGQ-AE 的边界中位 L2 跳变为块内中位的 21.25/23.04 倍，STATE 对照约 4.12 倍。原始模型动作到 WBC 双臂输入逐项最大误差为 0，真实 MuJoCo 关节随后反向；这已足以解释本样本的周期性顿挫，而非仅有视频容器错误。推理约 0.068–0.090 s，每步仿真固定推进 0.02 s；没有证据把录制视频中的反向归因于等待推理。状态/物理角差小于 1.2e-6 rad，输入图像随姿态变化。证据在运行根 `evals/stutter-diagnostic-20261104`，配置与命令在必读两桌 README。
- **诊断时钟的反例：** CPU 重放原生插值器，改成仿真时钟会使第 30 步尖峰看似变小，但下一步出现更大的完整跳变；必须查看边界前后整个窗口和真实运动，不能只挑边界一帧宣布修复。本轮未改控制时钟、模型动作、执行长度或资产，也未用平滑隐藏问题。MuJoCo qpos 为树遍历顺序，左右臂分别为 22:29/36:43，不能套 motor 的 body29 顺序使用 22:36；分析器已按官方模型核实，避免把手指错当右臂。图像更新和本样本映射一致不等于所有兼容问题均已排除；模型为什么重规划回到较早姿态仍需区分学习行为与其他输入/推理原因。原版场景资源替换按用户要求留到最后，未执行。
- **独立数值诊断初始化：** 首次额外对照采用 meta 构造后 assign 权重，但 rotary 非持久 buffer 不在 checkpoint，`.to(cuda)` 报 `Cannot copy out of meta tensor`；尚未开始对照推理，正式服务未受影响。实验脚本改为与服务相同的普通构造再严格加载，避免遗漏非持久 buffer；不能将此诊断脚本错误归为在线模型加载失败。首次日志保留为 `inference-mode-comparison-first-init.log`。
- **额外对照已完成：** 同一 VGQ-AE 权重、四张真实输入、相同 seed=17 噪声，eager / max-autotune 的双臂最大单关节差 0.004512 rad，最大联合 L2 差 0.009599 rad，存在 BF16 执行差异，不声称逐位等价。eager 在第 30 步也从实际左肩 -0.11909 rad 预测到 -0.03072 rad，仍会反向；本样本的大回跳不能以关闭编译解决。原始场景 `102344022` 与全部材质共 6014 文件，和 RHSCI 原资源 manifest 逐文件 SHA256 及文件名集合均一致，缺少/多出/不同为 0，未做无实际版本差别的资源替换。具体权重学习/其他兼容因素仍不能仅由一条截短轨迹全部排除；记录已证实的直接机制，不编造已修复的程序 bug。

- 本次静态审查及 CPU 复现发现等待队列的故障检测缺口：`experiments/simple_queue/run.py::current_complete()` 在检查 worker PID 前，遇到缺失的 `COMPLETE.json` 直接返回 False。如果当前 worker 意外死亡，队列会继续等待而非显式报错。当前 8 个 worker 均存活，尚未观察到此故障发生；已记录，未在仅要求检查的任务中改动故障处理流程。不能把一直等待当作评测仍健康运行，后续检查须同时查 PID 与官方统计进度。

- 单卡两套独立 π0.5 + Isaac 容量探针：GPU 2 上第二个 STATE 真实预热 `(30,36)` 通过，但场景加载峰值占用达到 30336 MiB，余量 1774 MiB；触发预设 2 GiB 保护，只结束额外 worker。未完成诊断 episode，不能称“双路正式评测通过”，也不能把保护触发说成已经 OOM。原 8 路继续运行，暂不扩大双路。额外 Isaac 初始化约 300 秒，原生 Kit 文件明确记录等待 `RtPso async group async compilation`；不能只凭控制台停在 app ready 就诊断死锁或 USD 错误。原始过程在 `evals/dual-gpu2-capacity-probe-20261104/`。
- 材质清单复核不能按目录外观擅自扁平化：官方 `vMaterials_2.zip` 自身同时包含顶层及嵌套 `vMaterials_2/` 成员，内外 `material_split.npy` 内容也不同。新核验脚本第一次擅自归一化成员路径，误报 2757 项多出及 split 冲突；已改为严格按 ZIP 原始成员路径比较。复核 50 个官方包、34068 个文件，缺少/多出/内容大小或 CRC 不符均为 0，88 处别名目录也均完全一致。没有因此移动或删除任何官方资源。前述“嵌套目录修复”指恢复加载器需要的顶层路径，不表示官方所有嵌套资源都应删除。
- 8 个 Isaac 默认各开数百线程，本机 112 CPU 的 load 曾超过 600；多路在 `_wait_for_viewport` 等待，不能只看 app ready 宣称开始 rollout。现按 NVIDIA 官方运行参数限制每实例 Carbonite/TBB 为 8，首轮 8 个客户端均已完成启动（部分约 190–200 秒），已有实际 30 步动作执行。此前尝试保留为 `-before-thread-limit`。线程限制是资源配置，不改控制频率/预测步数；未把所有等待都归因为已证明的缓存错误，辅助隔离缓存探针已停止，没有据未完成探针修改官方环境。
- Isaac `close()` 可能直接结束 Python，不能依赖关闭之后的结果写入，也不能只看退出码 0。实验编排已在每轮客户端退出后，核对官方 `stats/eval_stats.txt` 中 episode 0–9 恰好各一条，再写 `verified_results.json`；不足 10 条即失败，不写完成标记。补充核验前的一次预热输出保留在 `evals/pi05-between-30x30-5cycles-20261104-warmup-before-result-check`，尚未进入实际评测。
- vMaterials_2 曾被恢复脚本额外套一层目录，已调整至官方 `resolve_data_path` 期望的 data 根；归档和解压收据保留。仍需与上述 USD 子引用问题分开检查。
- 官方篮子 USD 的碰撞 mesh 没有 points，Isaac 会报不能建立该碰撞；本路径的物理由 MuJoCo 执行，但不能据此声称 USD 资产无错误。此问题不是材质缺失，未猜改官方资产。
- 官方 WBC/MuJoCo 手指映射历史问题、四项成功判定公开代码缺口仍按 `SIMPLE_UPDATES_20261103.md` 披露；两桌不属于那四个成功判定缺口任务。
- π0.5 保留历史 v2 分位归一化语义及其已知信息损失，不替换旧权重统计。推理无需 optimizer、RNG 或训练源码完全相同；不能为普通启动增加这些门槛。
