# RHSCI-Private 协作规则

本仓库为独立普通目录，当前分支 main；不创建工作树。修改前核对 status/HEAD，保留已有修改。

- 先读 [README](README.md)、[架构](docs/ARCHITECTURE.md)、对应 baselines 指南；涉及本次合入时读 [2026-10-30 整理记录](docs/PSI0_RAW_INTEGRATION_20261030.md)。后者记录本轮来源、差异、验收和兼容性，不代表 GPU 验收。
- 本机数据默认 `/mnt/cfs/cqizhh/personal/huangkailun/haoxuan/data`；权重、checkpoint 和日志默认同根 `checkpoints`。通用代码仍使用显式 dataset/output 参数，不硬编码本机路径；环境和缓存放代码目录之外。
- 原生骨干位于 src；课题代码放 src/rsci/backbones，公共数据与恢复逻辑放 src/rsci/data、src/rsci/runtime；入口放 scripts/train、scripts/data。不要复制历史 RSCI 目录或按日期的活动控制器作为新入口。
- 训练、恢复、推理使用对应骨干原生处理器和配方；ACT 无 Reader，EGQ-AE 不改称 VGQ-AE。π0.5 与 EgoVLA 的 Transformers 环境分离。
- 缓存仅保存原始 uint8 RGB/小张量；不保存增强或可训练特征。Guardian 验完整 SHA，worker 验清单/stat。完整断点含模型、优化器、调度器、RNG 和消费位置；契约变更拒绝原目录续训，不绕过校验或无点重训。
- 40k 训练保留 10k/20k/30k/40k 完整点，之间每 1k 滚动保存；新点验证后才能轮换旧滚动点。不长读活动滚动点，不操作 PSI0-RAW 训练、进程、契约或 checkpoint。
- 启动训练、完整评测、删除、移动或清理数据须有明确授权；整理源码不构成这些操作的授权。必要 CPU 检查可执行，实际覆盖与未验证范围分别报告。
- 关键新增逻辑写中文注释；文档登记到本文件及 README，常设规则变更同步工作区 AGENTS。新增来源应记录固定版本和逐文件 SHA，保留组件许可证。
- 经验证后仅 commit/push 本轮代码、配置、测试及文档；先检查 status 和 LFS。GitHub 不放权重、数据、环境或训练输出。权重使用公开 model ds1x/RHSCI/checkpoint，训练/评测数据使用 dataset ds1x/RHSCI 的 train-data/eval-data；发布前后核范围、类型、可见性、revision、大小和 SHA，保留原件。

2026-10-30 文档登记：main，[PSI0-RAW 整理记录](docs/PSI0_RAW_INTEGRATION_20261030.md)与[来源清单](docs/PSI0_RAW_INTEGRATION_20261030.json)；修改/迁移本批训练恢复和缓存前必读。CPU 验证范围见记录，CUDA/闭环及跨仓库 checkpoint 迁移未验。
