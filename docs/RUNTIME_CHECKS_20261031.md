# 运行检查精简（2026-10-31）

所属 RHSCI-Private/main，修改起点 `fb81ba0f78eb960b0c97a6283c89fb3d860277f0`。用户要求参考 Psi0 和 OpenWAM，取消不必要的检查和禁止。修改训练恢复、资产加载、推理启动或仓库规则前参考本记录。它替代历史整理记录中“源码任意变化必须换 run”“精确依赖锁与源码清单是运行门槛”“推理必须具备完整训练状态”的要求；不修改 PSI0-RAW 的生产队列。

## 参考与判断依据

阅读了本地 PSI0-RAW 的原生 Trainer，并在线核对 [Psi0 原生训练恢复](https://github.com/physical-superintelligence-lab/Psi0/blob/main/src/psi/trainers/trainer.py)及[部署入口](https://github.com/physical-superintelligence-lab/Psi0/blob/main/src/psi/deploy/psi0_serve_simple.py)。原生服务从配置构造模型并加载权重，检查设备、动作 horizon 和实际模型接口；不把训练优化器与原始源码树作为推理输入。

OpenWAM 存在同名项目，本轮同时查阅两个公开入口：[OpenWAM/OpenWAM 的 CLI](https://github.com/OpenWAM/OpenWAM/blob/main/src/open_wam/cli/train.py)让参数帮助在导入 Torch 训练栈前处理；[OpenWAM-Official/OpenWAM 的训练入口](https://github.com/OpenWAM-Official/OpenWAM/blob/main/scripts/train.py)由配置构造原生训练器，[部署脚本](https://github.com/OpenWAM-Official/OpenWAM/blob/main/scripts/deploy.sh)接收 checkpoint 目录并调用服务。这些是具体入口的对照，不推断它们全仓库没有校验，也不移植其算法或源码。网络查阅未下载、安装或修改依赖。

## 已取消或改为显式审计的项目

| 原行为 | 当前行为与原因 |
| --- | --- |
| 六模型训练契约绑定 Python/Shell 全量 SHA、依赖精确版本，重命名/注释也阻断恢复 | 比较模型、数据、优化器、调度与训练配置。旧收据中的源码/环境审计字段不参与比较；兼容时沿用已存身份及断点摘要，不改写旧记录 |
| π0.5 初始化文件时间戳进入契约 | 不以 mtime/ctime 判定实验兼容性；复制或触碰文件不等于模型发生变化 |
| ACT/EgoVLA/π0.5/GR00T/λ₀ 推理要求训练源码与当前 checkout 完全一致 | 默认不检查源码身份；按保存的配置加载权重并使用原生模型键/形状检查。π0.5/GR00T 的 `--training-source-root` 保留为显式来源审计 |
| 推理要求完整标记、优化器、scheduler、RNG、trainer state | 推理只检查模型配置及单文件/分片权重；训练状态文件可以不随推理模型交接。权重目录及文件允许 symlink |
| GR00T 硬编码 torch/Transformers/diffusers/torchcodec 精确版本与 FlashAttention 存在性；λ₀ 每次核完整环境锁 | 删除额外版本门槛及 λ₀ 无消费者的 `environment.py`；依赖是否支持所需 API 由实际导入和原生加载决定。版本锁仍是复现参考，移除门槛不等于证明任意版本兼容 |
| λ₀ prepare/train/serve 自动核对整个源码树与固定快照 | 取消自动源码扫描；组件来源清单与 `contracts.verify_upstream()` 保留作显式来源审计 |
| π0.5、EgoVLA、λ₀ 已有原生权重必须先生成 RHSCI 专用资产收据 | 接受没有专用收据的本地原生目录；收据存在时仍可记录来源，普通启动做文件/大小与原生加载检查，全量资产 SHA 用显式 `--check` / `--check-model` 或注册工具执行 |
| λ₀ 初始化硬绑定一个 Hugging Face repo/revision、精确文件集合 | 不限制来源仓库版本，也不因额外文件拒绝目录；保留模型所需文件检查与显式摘要审计 |
| 源码检查强制仅一个 worktree、禁止共享 Git 对象 | 删除与源码交付内容无关的限制；本轮仍在现有检出目录工作 |
| 每次开发检查都要求完整导出清单 SHA 不变 | `tools/check_public_export.py` 默认检查实际源码交付内容；`--verify-manifest` 在交接时显式核清单。许可证、凭据、模型/数据混入源码等检查保留 |
| AGENTS 强制反复阅读历史、堆积检查和禁止 | 缩为当前目录、命名、资产、授权、验证范围与本地提交规则；按任务查对应指南。用户“不自动推送远端”的约定保留 |

这次没有添加通用 `--skip-checks` 开关，也不把失败吞掉。取消的是不反映实际兼容性的门槛；用户无需靠绕过开关启动普通实验。

## 保留的检查

- 数据形状、动作维度/horizon、归一化、历史帧/episode 边界及 ARDY 对应关系。
- 模型参数键、张量形状和适用的数值有限性；缺失原生依赖仍会正常报错。
- 自动续训需要完整模型、Adam、scheduler、RNG 与数据消费位置；数值故障不盲目重试，无完整点不自动从零重训。
- 实验配置不同仍使用独立输出目录；进程锁防止重复写同一 run。新断点先完成和验收再轮换旧点。
- 清理范围、用户资产及其他活动任务的保护；本轮没有正式训练、完整评测、资产删除、环境安装或远端推送。

源码不再作为兼容性证明。修改算法语义但保持字段、形状不变时，系统不会自动判断科学可比性；需要独立实验与对应验证。这与原生按配置/权重工作的方式一致，不承诺跨代码或依赖版本逐位复现。

## 验证

CPU 套件使用既有独立环境，离线、`CUDA_VISIBLE_DEVICES=''`，分布式测试由 helper 排除。不以删除版本门槛替代 λ₀ 新 API 或 EgoVLA FlashAttention 依赖的安装和验收。

| 检查 | 实际结果 |
| --- | --- |
| Ψ0 / π0.5 / GR00T / ACT CPU 套件 | 分别43 / 37 / 32 / 29项通过 |
| EgoVLA | 28项通过；1项原生 SigLIP 导入仍因缺 `flash_attn` 失败 |
| λ₀ | 2项轻量测试通过；3个模型测试模块因现有 Transformers 4.53.2 缺 `Qwen3_5Config` 无法导入 |
| 公共入口、配置兼容、守护、完整状态、推理、官方客户端 | 35项通过，与骨干套件有重叠，不累加为唯一测试数 |
| ACT 原生 HTTP 序列化 | 单独执行1项通过 |
| 六模型模块及六个 shell 入口 | 真实 `--help` 全部退出0，无训练启动 |
| 静态与交接 | 163文件 Ruff 检查/格式、890 Python语法、13 shell语法、文档文件链接通过；组件及总清单显式SHA核验通过 |

验证环境沿用[模型布局记录](NATIVE_PACKAGE_LAYOUT_20261031.md)中的本机隔离环境，没有安装依赖。Git LFS工具未安装；源码交接扫描没有发现 LFS指针或运行资产混入。CUDA/完整多卡训练、恢复旧大模型点和 SIMPLE闭环未执行。

新增回归覆盖：旧源码/依赖记录变化时复用已保存断点身份；模型、数据和学习率变化仍拒绝；没有私有资产收据的原生目录；推理无需优化器/RNG/训练标记；历史来源仅显式审计；软链接模型目录可用；已有完整断点和真实守护测试继续执行。
