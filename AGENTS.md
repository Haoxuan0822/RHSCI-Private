# RHSCI-Private 协作规则

- 修改前查看 Git 状态并保留已有改动；默认在当前检出目录工作。
- 按任务查阅 [README](README.md)、[架构](docs/ARCHITECTURE.md)及对应 `baselines/<model>/README.md`，无需每次重复读取全部历史记录。首次公开克隆安装/训练按[全新环境步骤](docs/TRAINING.md#from-a-fresh-clone)和[现行资产交接](docs/ASSETS.md#current-rhsci-handoff)，历史备份路径仅供溯源。
- 项目名称统一为 RHSCI，当前 Python 包/模块名为 `rhsci`，类名采用 `RHSCI`。旧权重键、配置字段和产物文件名保留兼容。六个模型包统一放在 `src/<原生包名>/`，扩展放在包内 `rhsci/`；`src/rhsci` 放跨模型公共工具。沿用对应骨干的处理器、算法和配方；ACT 使用 EGQ-AE，无 Reader。
- 可复用的算法和基础设施优先放公共层，模型只声明实际差异；已共享函数见[架构表](docs/ARCHITECTURE.md#shared-functions-2026-11-01)。函数用明确的动作与对象命名，同类职责保持一致；关键逻辑用中文说明输入、返回值及副作用。目录与命名见[指南](docs/DIRECTORY_STRUCTURE.md)。
- 解释架构、迁移后续 PSI0-RAW 方法或宣称复现前，阅读[2026-11-02 函数与复现审查](docs/ARCHITECTURE.md#function-audit-20261102)。当前 Ψ0 VGQ-VLM 是 Reader 后二次语言前向，不能当作 PSI0-RAW 后续四原图直接 VLM Query；GR00T 初始化与固定 SIMPLE 预设不同。CPU 回归不代表官方全模型等价或 GPU 闭环通过。
- 数据、权重、环境和训练输出放在检出目录外。此机器默认数据根 `/mnt/cfs/cqizhh/personal/huangkailun/haoxuan/data`，权重与日志为同根 `checkpoints`；显式路径优先。
- 缓存保存原始 RGB/小张量。续训核对模型、数据、配方和完整训练状态；源码摘要、注释、重命名和依赖精确版本不作为启动门槛。旧收据可保留作记录。
- 推理加载配置、处理器/归一化及模型权重，不要求优化器、调度器、RNG、训练源码快照或完整训练标记。资产清单与全量 SHA 审计用于显式核验。
- 保留数值异常报错、模型键/维度校验、进程锁和新断点写完后再轮换旧点的保护。正式训练、完整评测和数据清理按用户授权执行；不操作 PSI0-RAW 的活动训练、进程或断点。
- 验证覆盖实际修改及相关调用，报告环境限制；无需为普通改动重复全量检查。源码清单、来源与许可证用于交接，更新涉及的条目即可。
- 文档直接维护对应指南及入口；跨仓库规则变更只同步适用范围，不复制运行流水。
- 仅本地提交本轮改动。用户明确要求 **不自动推送远端**；只有收到推送指令才执行远端发布。

当前结构与历史来源：[原生包整理及可读性记录](docs/NATIVE_PACKAGE_LAYOUT_20261031.md)、[初次整合](docs/PSI0_RAW_INTEGRATION_20261030.md)、[训练职责](docs/TRAINING_STRUCTURE_20261030.md)、[官方 SIMPLE](docs/SIMPLE_OFFICIAL_20261030.md)。历史记录描述当时行为，2026-10-31起运行校验以[精简记录](docs/RUNTIME_CHECKS_20261031.md)及现行使用指南为准；修改恢复、部署和资产检查前参考该记录。
