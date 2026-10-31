# 按原生模型包聚合代码（2026-10-31）

2026-11-01：下文目录表为迁移当天记录；当前六模型已平铺到 `src/<原生包名>/rsci`，以[现行目录说明](DIRECTORY_STRUCTURE.md#package-depth-2026-11-01)和[公共函数表](ARCHITECTURE.md#shared-functions-2026-11-01)为准。

后续用户要求精简运行检查：本文关于源码 SHA 任意变化阻断恢复/推理的历史行为已被[运行检查精简](RUNTIME_CHECKS_20261031.md)覆盖。模块命名记录继续适用；文件位置以现行目录为准。

所属：RHSCI-Private/main 普通目录，修改起点 `2199f9207054ed5f94feca4b94e4c6bfaebe1290`。用户确认沿用 Psi0 原有包名，将同一模型的专属扩展放回对应包，`src/rsci` 仅保留跨模型工具。修改模型入口、源码契约、环境导入或使用历史模块路径前必读。本轮本地提交，不推送远端，不创建工作树。

## 实际布局

| 模型 | 当前扩展源码 | 当前模块前缀 |
| --- | --- | --- |
| Ψ0 | `src/psi/rsci/` | `psi.rsci` |
| π0.5 | `src/openpi/rsci/` | `openpi.rsci` |
| GR00T | `src/gr00t/gr00t/rsci/` | `gr00t.rsci` |
| EgoVLA | `src/egovla/human_plan/rsci/` | `human_plan.rsci` |
| λ₀ | `src/openlambda/lambda0/lambda0/rsci/` | `lambda0.rsci` |
| ACT | `src/act/rsci/` | `act.rsci` |

原生目录及包名保持。每个原生包内的 `rsci/` 存放该模型的模型扩展、数据/处理器、训练配方、Trainer、守护、完整断点和推理入口；相邻的原生实现继续直接复用。`src/rsci/backbones` 及其包初始化文件已移除，不保留第二套源码、旧别名或兼容转发包。公共 `rsci.data`、`rsci.runtime`、`rsci.inference` 和路径工具保留。

`config/preflight/contract/trainer/worker/guardian` 的职责不变；Ψ0/λ₀ 保持各自原生 Trainer/backend 接入方式。模型、损失、增强、归一化、动作维度、学习率、优化器、调度、批量、Reader、checkpoint schema 与缓存规则没有新增配方变更。按原生包聚合不等于所有模型共享一套配方。

## 调用链和定位

公开入口仍为 `scripts/train/{psi0,pi05,gr00t,egovla,lambda0,act_rsci}.sh`；脚本内的模块名、子进程递归、native hooks、推理调用和测试 mock 路径同步更新。例：`pi05.sh → openpi.rsci.train → guardian/worker → 原生 openpi 模型 → trainer/checkpoints`。共享 HTTP 入口 `rsci.inference.http_server` 按骨干调用 `openpi.rsci.serve` 或 `gr00t.rsci.serve`；其他服务入口见[推理文档](INFERENCE.md)。

`rsci.paths.CODE_ROOT` 从公共文件位置确定仓库根，替代迁移模块中依赖固定 `parents[4]` 的定位。`BACKBONE_MODULES` 仅登记六个模块位置，导入登记表不加载模型或可选依赖。λ₀ 源码、共享工具、环境锁和 patch 的契约定位同步更新，不因目录加深而绑定错误文件。

GR00T、EgoVLA、λ₀ 的原生 Python 包根分别为 `src/gr00t`、`src/egovla`、`src/openlambda/lambda0`；对应 shell 启动器继续设置各自 `PYTHONPATH`。直接执行模块时也需要该模型环境及包根，不能只把其外层源码目录名当成 Python 包名。EgoVLA 实际包名是 `human_plan`，其 VILA 仍独立位于 `src/egovla/VILA`。Ψ0/π0.5/ACT 包根为 `src`。

## 身份、来源与旧产物

新源码契约使用当前模型文件路径和 SHA，旧训练目录不通过新契约恢复；不修改旧 run_config、launch_contract、完整标记、优化器或 checkpoint。需要恢复旧训练时使用匹配的旧 Git 版本和独立环境，不能绕过保护或无点重训。既有显式历史 π0.5 推理来源检查保留，CPU 回归通过，不由此宣称任意历史模型均可在新目录恢复。

首次整合的 JSON SHA 清单保持历史原样，2026-10-30 两份迁移/训练职责记录保留旧提交路径并增加现行入口。当前文件清单由根 `PUBLIC_EXPORT_MANIFEST.json` 记录。λ₀ 的完整组件清单加入本地迁入扩展并明确其为 RHSCI 代码；GR00T 清单登记迁入扩展及原有 N1.6 衍生限制，不将研究扩展冒称官方原件。许可证和原始来源提交不变。

## 实际验证

验证机器为本机，工作目录 `/root/workspace/RHSCI-Private`。CPU 测试仅使用已有环境、微型模型和临时数据，`CUDA_VISIBLE_DEVICES=''`、离线模式与禁止 pyc；没有正式训练、评测、下载、安装、镜像制作或迁移资产。既有 PSI0-RAW 活动源码、环境、配方、进程和断点保持不变。

| 检查 | 实际结果 |
| --- | --- |
| Ψ0 CPU 套件 | 43 项通过 |
| π0.5 CPU 套件 | 37 项通过 |
| GR00T CPU 套件 | 32 项通过 |
| ACT CPU 套件 | 29 项通过 |
| EgoVLA CPU 套件 | 29 项中 28 项通过；SigLIP 导入因既有环境缺 `flash_attn` 失败 |
| λ₀ CPU 套件尝试 | 2 项轻量测试通过；3 个模型测试模块因环境 Transformers 4.53.2 缺 `Qwen3_5Config` 导入失败，未声称完整套件通过 |
| 共用入口/新包定位/守护/完整断点/HTTP/官方客户端 | 31 项通过；与上述骨干套件存在覆盖重叠，不直接相加为唯一测试数 |
| 六模型模块与六个公开 shell 入口 | 实际 `--help` 均退出 0，不创建 worker 或正式 run |
| 迁移行为核对 | 归一化新旧模块路径后，164 个顶层函数/类 AST 一致；其余10个仅源码/资产定位、共享契约路径或导入排序改变，逐项差异已核对 |
| Python/Shell 语法 | 888 个 Python 文件解析、15 个公开 shell 语法通过 |

CPU helper 继续排除分布式测试；没有验证全尺寸 CUDA、实际多卡训练、吞吐或 SIMPLE 闭环。SIMPLE 外部源码、9 个子模块、版本锁和 LFS/环境待补状态保持既有记录，不将本次布局调整称为完成仿真部署。

重跑各骨干测试使用 `PYTHONDONTWRITEBYTECODE=1 <对应环境>/bin/python -B tools/test_cpu.py --backbone <名称>`。Ψ0/ACT 使用 `/root/workspace/PSI0-RAW/.venv-psi/bin/python`；π0.5/GR00T/EgoVLA 使用用户 CFS data 下 `.envs/{pi05-rsci,gr00t-rsci,egovla-rsci}/bin/python`。λ₀ 本次在 EgoVLA 环境做兼容性尝试，不能用于正式 λ₀ 训练。

守护测试实际启动并终止自建的空闲子进程，保留其他任务。复验时发现旧测试 fixture 未传显式缓存配置且对 EgoVLA 入口重导出打补丁，而实际实现已在 guardian 模块；修正为完整 fixture 并 patch 实际 guardian，真实 SIGTERM 转发/回收通过，未改变生产守护行为。

维护代码 Ruff 检查显式使用根配置：`ruff check --config ruff.toml .` 与 `ruff format --config ruff.toml --check .`，避免 GR00T 原生子项目配置覆盖公共规范；新位置已纳入根检查范围。完整源码清单、组件清单和文档链接在交付前核验。环境依赖缺口保留，不为让测试通过而修改现有训练环境。

## 可读性与函数命名整理（2026-10-31）

所属 RHSCI-Private/main，本次起点 `e0f5302c17846d9ec3bdd371604ef8cb32ffd586`。用户要求代码与函数名更易读；修改扩展训练/恢复入口、直接 Python 导入及推理应用工厂前必读。当前名称与阅读顺序见[目录指南](DIRECTORY_STRUCTURE.md#function-names-and-reading-order)。

六个模型将泛化简称改为明确动作：`parser → build_argument_parser`，四个 HF 接入的 `resolve → resolve_training_paths`，`contract → build_training_contract`，`train → run_training_worker`，`guardian/guard → run_training_guardian`，`trainer_type/trainer_class → create_trainer_class`，断点回调统一 `build_checkpoint_callback`。Ψ0 的配置/命令组合使用 `resolve_training_plan`；λ₀ 使用 `build_training_config`、`validate_training_data`、`validate_output_directory`、`execute_training`。这些是函数重命名，模块名与 shell 启动入口保持。调用、直接导入、动态导入后的属性、mock 目标、嵌入的子进程代码均已同步，没有添加旧名称别名。

55 个关键函数补充中文说明，写明返回对象、只读检查或实际训练等副作用；CLI 构建器的 `p/result` 改为 `argument_parser`。GR00T 的数据预检和 EgoVLA 的完整模型/spawn worker 一致性检查从 `train.main` 提取到 `preflight.py`，原检查内容与执行时机保留。EgoVLA `main` 从101行缩短为43行，GR00T 从111行缩短为80行。共享启动器使用 `build_launch_plan`，HTTP 参数构建使用 `build_argument_parser`，ACT HTTP 工厂使用 `create_app`。框架 `forward/train`、Trainer 钩子、checkpoint 字段与 CLI 参数没有重命名，官方 SIMPLE 与相邻原生模型实现没有改写。

实际验证在上述同机、同独立环境及离线 CPU 范围完成：Ψ0 43、π0.5 37、GR00T 32、ACT 29 项通过；EgoVLA 29项中28通过，1项仍因缺 `flash_attn` 失败；λ₀ 2项轻量测试通过、3个模型测试模块仍因 Transformers 4.53.2 缺 `Qwen3_5Config` 无法导入。公共入口/守护/完整断点/HTTP/官方客户端31项通过，另外单独执行 ACT 原生序列化 HTTP 测试1项通过；各组存在重叠，不合并宣称唯一测试总数。六个实际模块和六个 shell 的 `--help` 均退出0。没有安装或下载依赖、启动正式训练、GPU/多卡或 SIMPLE 闭环。

39个修改的源码/脚本文件进行 AST 对比：归一化函数/局部名称、文档字符串和导入，并将两段提取的预检代码放回原位置后，执行语句结构一致；导入和实际入口另由 CPU 测试及 `--help` 检查覆盖。这是重构核对，不等于全尺寸运行验证。维护范围161个文件 Ruff 检查/格式通过，组件与总清单 SHA 同步更新。

直接导入旧 Python 简称的外部脚本需更新到当前名称。源码 SHA 因此改变，旧 run 仍须使用匹配 Git 版本恢复；不修改旧契约或绕过不兼容保护。本轮只本地提交，不推送；PSI0-RAW 的活动源码、配方、环境、进程与断点保持。
