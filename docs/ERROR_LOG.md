# 错误记录

每次开始工作前必须完整阅读本文件和 `AGENTS.md`。错误按实际证据记录，不用静态检查、下载成功或进程存活替代实际运行验证；修复后更新原条目中的状态与验证范围。

## USD 引用旧路径失效：已修复，四难度各首个初态渲染通过

- 现象：实际 Isaac 渲染日志出现 `Could not open asset @/root/tmp/bdb1f316aa0188ec/props/TV_Screen2.usd@`，同时还有 Shader11、MIRROR 等 USD 子资源引用错误。不能把它归为不影响场景的普通启动噪声。
- 原因：官方场景 USD 内含 `/root/tmp/<编号>/props` 等旧绝对路径；本机该路径仍是指向已删除旧运行目录的失效软链接。官方加载器准备的 `/tmp/<编号>/props` 及当前官方源资源实际存在，但不会自动修正 `/root/tmp` 下的旧链接。
- 遗漏：此前修复只扫描 `/tmp`，没有覆盖 `/root/tmp`；默认日志级别还会省略部分警告。材质文件存在、烟雾测试能出图，不代表 USD 组合完整。
- 处理：`experiments/pi05_between/repair_usd_paths.py` 扫描官方 HSSD USD 中实际引用的 `/tmp`、`/root/tmp` 路径；仅将失效链接改名保留，再建立指向当前官方源资源的兼容链接，不改官方 USD 文件。正式评测客户端开启 warning 级别保留引用警告。
- 验证：88 处别名均存在；四难度各 episode 0 完成真实 reset/render，逐 prim composition errors 全为 0、图像均为有效 `(360, 640, 3)`。`logs/pi05-between-material-rendering-textures-restored.log` 中 `Could not open asset`、缺失纹理、`UsdToMdl`、`Failed to resolve`、traceback 全为 0；`logs/pi05-between-material-rendering/result.json` 保存结果，level-2 图像已人工查看。范围是四个初态的实际渲染与全部 40 初态的静态 MDL 检查，不宣称所有后续 rollout 已完成；后续仍检查正式日志。
- 防止复发：扫描资产中真实嵌入的所有绝对路径，而不是只猜一个临时根；同时检查 MDL 文件、导出名称、纹理、USD 子引用与实际图像。保留首次失败日志及旧链接，明确验证覆盖哪些初态。
- 补充发现：USD crate 的引用可能被压缩，`strings` 返回零条不能证明没有引用；实际相对路径可解析至 `/root/tmp`。修复脚本已改用 `Sdf.Layer.GetExternalReferences()`，解析并恢复了 44 处场景引用别名。渲染诊断使用了 Isaac Sim 4.5 未提供的 `Stage.GetCompositionErrors()`，又因 Kit 的 `close()` 掩盖 traceback，导致初次看不到失败原因。现改为逐 prim 读取 `GetPrimIndex().localErrors`，先保存结果或 traceback，再关闭；不要拿其他 USD 版本的 API 当作本环境事实。
- 为避免错误场景计入评测，已主动停止首次 8 卡启动，并将完整输出保留为 `evals/pi05-between-30x30-5cycles-20261104-before-usd-repair`。这次尝试不计入正式 400 次。
- 分层复验继续发现 `../textures/*.png` 缺失：只恢复 props 会使 USD composition errors 清零，却仍缺背景纹理。修复已改为同时恢复每组 props 与 textures；必须再看 `[UsdToMdl]` 和资产解析日志，不能只用 composition errors 判定材质正常。

## RTX 5090 首次 π0.5 编译候选内核不支持：真实推理已通过

- 现象：Torch 2.7 / Triton 3.3 的 max-autotune 预热中，部分候选报 `computeCapability not supported`、`PassManager::run failed`，日志显示 cc=120。
- 已知范围：原生模型严格加载通过；编译调用仍在执行，栈检查确认在图处理/代码生成中，并非仅凭日志推断卡死。候选失败不等于整个推理失败，也不能在实际 `/act` 返回前称兼容验证通过。
- 处理：保留原推理公式与编译设置，完成真实请求验证；没有通过删日志、换采样步数或假模型掩盖错误。
- 后续证据：8 个模型均实际返回有限的 `(30, 36)`，原生首次编译预热通过，无需切换编译 backend。候选内核失败与完整推理失败必须分开判断。

## 其他仍需保留的已知限制

- 单卡两套独立 π0.5 + Isaac 容量探针：GPU 2 上第二个 STATE 真实预热 `(30,36)` 通过，但场景加载峰值占用达到 30336 MiB，余量 1774 MiB；触发预设 2 GiB 保护，只结束额外 worker。未完成诊断 episode，不能称“双路正式评测通过”，也不能把保护触发说成已经 OOM。原 8 路继续运行，暂不扩大双路。额外 Isaac 初始化约 300 秒，原生 Kit 文件明确记录等待 `RtPso async group async compilation`；不能只凭控制台停在 app ready 就诊断死锁或 USD 错误。原始过程在 `evals/dual-gpu2-capacity-probe-20261104/`。
- 材质清单复核不能按目录外观擅自扁平化：官方 `vMaterials_2.zip` 自身同时包含顶层及嵌套 `vMaterials_2/` 成员，内外 `material_split.npy` 内容也不同。新核验脚本第一次擅自归一化成员路径，误报 2757 项多出及 split 冲突；已改为严格按 ZIP 原始成员路径比较。复核 50 个官方包、34068 个文件，缺少/多出/内容大小或 CRC 不符均为 0，88 处别名目录也均完全一致。没有因此移动或删除任何官方资源。前述“嵌套目录修复”指恢复加载器需要的顶层路径，不表示官方所有嵌套资源都应删除。
- 8 个 Isaac 默认各开数百线程，本机 112 CPU 的 load 曾超过 600；多路在 `_wait_for_viewport` 等待，不能只看 app ready 宣称开始 rollout。现按 NVIDIA 官方运行参数限制每实例 Carbonite/TBB 为 8，首轮 8 个客户端均已完成启动（部分约 190–200 秒），已有实际 30 步动作执行。此前尝试保留为 `-before-thread-limit`。线程限制是资源配置，不改控制频率/预测步数；未把所有等待都归因为已证明的缓存错误，辅助隔离缓存探针已停止，没有据未完成探针修改官方环境。
- Isaac `close()` 可能直接结束 Python，不能依赖关闭之后的结果写入，也不能只看退出码 0。实验编排已在每轮客户端退出后，核对官方 `stats/eval_stats.txt` 中 episode 0–9 恰好各一条，再写 `verified_results.json`；不足 10 条即失败，不写完成标记。补充核验前的一次预热输出保留在 `evals/pi05-between-30x30-5cycles-20261104-warmup-before-result-check`，尚未进入实际评测。
- vMaterials_2 曾被恢复脚本额外套一层目录，已调整至官方 `resolve_data_path` 期望的 data 根；归档和解压收据保留。仍需与上述 USD 子引用问题分开检查。
- 官方篮子 USD 的碰撞 mesh 没有 points，Isaac 会报不能建立该碰撞；本路径的物理由 MuJoCo 执行，但不能据此声称 USD 资产无错误。此问题不是材质缺失，未猜改官方资产。
- 官方 WBC/MuJoCo 手指映射历史问题、四项成功判定公开代码缺口仍按 `SIMPLE_UPDATES_20261103.md` 披露；两桌不属于那四个成功判定缺口任务。
- π0.5 保留历史 v2 分位归一化语义及其已知信息损失，不替换旧权重统计。推理无需 optimizer、RNG 或训练源码完全相同；不能为普通启动增加这些门槛。
