# Code organization

The project and current Python namespace are **RHSCI / `rhsci`**. Model extensions use `psi.rhsci`, `openpi.rhsci`, `gr00t.rhsci`, `human_plan.rhsci`, `lambda0.rhsci`, `act.rhsci`; common tools use `rhsci`. Current CLI mode is `--mode rhsci`. `RHSCI_*` environment names take priority; historical `RSCI_*` names and `--mode rsci` are accepted as compatibility inputs. Existing checkpoint tensor/config keys, schema strings, filenames such as `rsci_config.json` / `RSCI_COMPLETE.json`, and runtime/environment directories retain their saved spelling. This avoids changing model parameter identities or old artifact locations. Historical source records describe their original names.

Model-specific RHSCI extensions live inside their native packages. `src/rhsci` contains shared utilities; importing its root does not load a backbone or optional dependencies. See [Repository structure](DIRECTORY_STRUCTURE.md) for the directory tree and per-backbone entry points.

Use descriptive action names for maintained extension functions, consistently across models. The [function guide](DIRECTORY_STRUCTURE.md#function-names-and-reading-order) maps CLI parsing, path resolution, preflight, contracts, workers and guardians to their owning modules. Keep detailed preflight checks in `preflight.py` and state return values and side effects in critical function docstrings. Native framework method names, CLI options and persisted checkpoint keys follow their original interfaces.

| Directory | Responsibility |
| --- | --- |
| `src/psi/rhsci`, `src/openpi/rhsci`, `src/gr00t/rhsci`, `src/human_plan/rhsci`, `src/lambda0/rhsci`, `src/act/rhsci` | Model-specific integration, data, training, checkpoints and serving, colocated with each native implementation |
| `src/rhsci/data` | ARDY sidecars, dataset provenance, raw LeRobot input and frame caching |
| `src/rhsci/runtime` | Artifact writes, checkpoint retries, numerical checks, process cleanup and profiling |
| `src/rhsci/inference` | SIMPLE wire format, HTTP app, control-step RGB history and serving utilities |
| `scripts/train` | Common launcher, shell entry points and distributed launch configuration |
| `baselines/<backbone>` | Usage, dependencies and component integration records |
| `tests` | Model, data, resume, numerical failure, serving and launcher regressions |

Native packages such as `psi`, `openpi`, `gr00t`, `egovla` and `openlambda` retain their source trees under `src`. Each backbone uses its own dependency environment. Shared utilities belong in `src/rhsci`; native model dependencies belong in the corresponding adapter.

PSI0 integrates through `RHSCIFinetuneTrainer` and native trainer hooks. λ₀ integrates through `RHSCITrainingBackend` and explicit server arguments. ACT, π0.5, GR00T and EgoVLA share sampler-epoch recovery through `runtime/hf_resume.py`; its compatibility check stays local to each derived Trainer. ACT's bounded decoder/frame cache stays in its adapter because it depends on its native LeRobot pipeline. Being-H0.5 retains official code under `src/beingh05` and uses only the native common launcher. The external SIMPLE client integration is an opt-in wrapper; see [Inference](INFERENCE.md).

ACT, EgoVLA, π0.5 and GR00T share the complete-checkpoint lifecycle in `runtime/hf_full_state.py`. ACT/EgoVLA bind their own schema and model loader; π0.5/GR00T bind the existing schema-2 format and their distinct metadata files. Ψ0 and λ₀ keep their native checkpoint integration. Contracts compare model, data and training configuration; source/version audit records are excluded from runtime compatibility. Inference validates model files independently of optimizer/RNG state; see [runtime check policy](RUNTIME_CHECKS_20261031.md).

## Shared functions (2026-11-01)

| Shared implementation | Consumers | Per-model responsibility |
| --- | --- | --- |
| `runtime/losses.py`: `masked_auxiliary_loss`, `masked_ardy_loss` | All six backbones | Native action loss, auxiliary head and feature construction |
| `data/sampling.py`: `EpochRandomSampler`, `EpochSampler` | Ψ0, ACT, GR00T | Whether samples carry explicit `(epoch,index)` |
| `runtime/contracts.py`: `compute_contract_digest` | ACT, EgoVLA, π0.5, GR00T, λ₀ | Model/data/recipe fields; old digest serialization is preserved |
| `runtime/hf_full_state.py`: `HFCheckpointProtocol`, `HFTrainerCheckpointProtocol` | ACT, EgoVLA, π0.5, GR00T | Existing schema, metadata files, native parameter loading |
| `runtime/hf_callbacks.py`: `build_checkpoint_callback` | ACT, EgoVLA | Protocol binding; π/GR keep their processor/config save hooks |
| `runtime/numerics.py`: `require_finite` | ACT, EgoVLA, π0.5, GR00T | Which losses/gradients are checked and when |

Native Reader layouts, model execution, normalization, action losses, optimizer/scheduler recipes and specialized resume handling remain with each backbone. Common functions are imported or bound directly; identical algorithms are not copied into each adapter. The shared loss masks invalid targets, preserves a zero-gradient graph when all labels are invalid, and normalizes uneven DDP ranks by global valid sample count. Valid nonfinite labels fail explicitly.

Validation for this refactor uses the existing CPU backbone suites, a two-rank CPU/Gloo loss-versus-centralized-gradient comparison, empty-label gradients, epoch/RNG independence, schema-2 sharded checkpoint compatibility, native package resolution and real CLI help. Actual results from `fdd7edf` plus this refactor: Ψ0 43, π0.5 37, GR00T 32, ACT 29 CPU tests passed. Shared suite: 37 passed and 2 SIMPLE tests initially skipped without `SIMPLE_HOME`; those 2 were then run against the existing pinned checkout and passed. The four new shared tests include the real two-rank CPU/Gloo comparison. λ₀ source-audit tests: 2 passed, including relocated-package changes. All 12 module/shell training help calls passed. Python/shell syntax, Ruff, package discovery, source inventories and documentation links were checked without installation. Counts overlap across suites and are not a unique-test total. Full GPU training/old large-model resume remain untested. EgoVLA's existing missing FlashAttention and λ₀'s missing Qwen3.5 API remain environment limitations.


<a id="function-audit-20261102"></a>

## 函数调用链与复现审查（2026-11-02）

本节按 RHSCI-Private `78a100d` 的实际调用审查，更新上述历史验证记录的当前适用边界。结论：包目录已统一，共用损失与完整断点基础设施已落地，但训练守护和推理接口仍分叉；不能将当前仓库整体称为“严格复现官方实验”或“与最新 PSI0-RAW 完全一致”。本轮仅检查源码、运行现有 CPU 回归并更正文档，不改变模型算法或现有训练。

### 入口与训练职责

```text
scripts/train/baseline.py::build_launch_plan()       可选公共入口：选择模式、环境与脚本
    └─ scripts/train/<backbone>.sh                  也可直接调用，设置各骨干环境
         └─ <native_package>.rhsci.train/launch::main()
              ├─ 参数与路径 → 数据/资产检查 → 训练契约
              ├─ guardian：持锁、选择完整点、启动/等待 worker
              └─ worker：构建数据/模型/Trainer → 正式训练
```

这是一套共同职责，并非六个模型都调用同一个总控制器。`--check-data`、`--check`、`--dry-run` 分支按骨干定义，也不是所有模式都执行上面的完整链路。

| 骨干 | 实际函数入口与训练链路 | 模型和数据的所有者 |
| --- | --- | --- |
| Ψ0 | `psi.rhsci.train.main` → `resolve_training_plan` / `run_training_preflight` / `build_training_contract` → `run_training_guardian` → 原生 `scripts/train.py::train` → `RHSCIFinetuneTrainer` | `psi.rhsci.model` 扩展方法；原生 `psi.models.psi0`、`psi.trainers.finetune` 和 LeRobot 数据层保留接入钩子 |
| π0.5 | `openpi.rhsci.train.main` → `resolve_training_paths` / `run_training_preflight` / `build_training_contract` → `guardian.run_training_guardian` → `worker.run_training_worker` → `trainer.create_trainer_class` | `openpi.rhsci.data` / `processing` / `model`；底层复用原生 PaliGemma 与 Action Expert 模块 |
| GR00T | `gr00t.rhsci.train.main` → `resolve_training_paths` / `run_training_preflight` / `build_training_contract` → `guardian.run_training_guardian` → `worker.run_training_worker` → `trainer.create_trainer_class` | `gr00t.rhsci.data` / `config`；研究前向和 `VisualStateQuery` 仍位于原生 `gr00t.model.gr00t_n1d6` |
| EgoVLA | `human_plan.rhsci.launch.main` → `train.main` → `resolve_training_paths` / `check_data` / `assets` / `build_training_contract` → `guardian.run_training_guardian` → `worker.run_training_worker` → `trainer.create_trainer_class` | `human_plan.rhsci.data` / `processing` / `model`；VILA/LLM 依赖保留独立环境 |
| λ₀ | `lambda0.rhsci.launch.main` / `run_training_guardian` → `train.main` → `build_training_config` / `validate_training_data` → `execute_training` → 原生引擎＋`RHSCITrainingBackend` | `lambda0.rhsci.model` / `data` / `runtime`；后端负责原生模型、优化器、数据游标和保存/恢复的适配 |
| ACT | `act.rhsci.launch.main` → `train.main` → `resolve_training_paths` → `launch.run_training_guardian` → `worker.run_training_worker` → `trainer.create_trainer_class`；检查分支使用 `check_data` | `act.rhsci.data` / `processing` / `model`；原生 LeRobot ACT 执行动作分支 |

四个 HF 骨干的 `config.py` 负责参数、路径和 `TrainingArguments`，`preflight.py` 负责数据/资产检查，`contract.py` 记录模型/数据/配方，`worker.py` 组装训练对象，`trainer.py` 定义优化器、损失、回调及恢复适配。`train.py` 目前仍重导出一批旧函数，阅读实现应进入实际所属模块。

### 模型前向，精确到核心函数

| 骨干 | 核心函数与数据流 | 推理路径 |
| --- | --- | --- |
| Ψ0 | `RHSCIPolicyMixin._rhsci_forward` 取得当前 VLM 特征并调用原生动作头；`_rhsci_visual_memory` 编码历史帧；`RHSCIQuery.memory` 形成四帧 memory；`_rhsci_query_latent` 复用 AE blocks 或再次调用语言 Transformer；辅助预测进入 ARDY 损失 | `_rhsci_inference_features`：无星号方法跳过 Query/历史；带星号方法将 latent 加入动作条件，`rhsci_history_from_request` 要求真实历史协议 |
| π0.5 | `Pi05RHSCI.forward` → `encode` → `memory_cache` / `expert` 计算 flow 动作损失；`encode_history` → `Reader.forward` → 共享 `expert` 计算 Query 辅助损失 | `sample_actions` 执行 10 步去噪；`serve.Server.predict_action` 反归一化 30×36，再返回前 24 步；不执行辅助支路 |
| GR00T | 原生 `Gr00tN1d6` 前向中接入 `VisualStateQuery.memory` / `forward`；每帧图像/文本 Reader 适配原生 DiT 交替注意力，Query 复用 DiT，再加辅助损失 | `rhsci.serve.build_server` 绑定 HTTP 服务；动作使用原生模型路径，辅助监督不作为部署输入 |
| EgoVLA | `EgoVLARHSCI.forward` → `_encode` → `action_prediction` / `decode_latents` 计算动作；历史编码 → `Reader.forward` → 同一 decoder 的 Query → ARDY | `sample_actions` 只执行动作路径；`serve.predict` 处理当前观测与响应 |
| λ₀ | `Lambda0RHSCI.forward_vlm` 捕获当前特征；`forward` 保留父类动作前向并调用 `auxiliary` → `HistoryReader.forward` → 共享 expert blocks → ARDY | 原生推理链路，保留其 RTC；服务入口为 `lambda0.rhsci.serve.main` |
| ACT | `RHSCIACT.forward` → 原生 `policy.model`，保留 masked L1＋10×KL；`auxiliary_prediction` 用原生 ResNet 的四帧特征、共享 encoder/decoder Query 加 0.1×ARDY | `predict_action` / `select_action` / `reset`；latent=0，输出 16×36；EGQ-AE 没有 Reader 或语言模型 |

四帧历史按 50 Hz 的 `[-6,-4,-2,0]` 取样；标签来自 sidecar，不能把监督标签当动作条件。上述为当前各方法的主干职责，不表示不同 Reader、注意力或动作损失可互换。

### 已复用与仍然分叉的基础设施

- `rhsci.data.ardy_sidecar.ArdySidecarReader` 读取 ARDY 与有效位；`RawInputCache` 读取原始 uint8/小张量；`EpochRandomSampler` / `EpochSampler` 提供 epoch 采样。增强和特征在训练时计算。
- `rhsci.runtime.losses.masked_auxiliary_loss` / `masked_ardy_loss` 统一有效标签与 DDP 全局有效数归一化；各骨干保留各自动作损失。
- `runtime.contracts.training_configuration` / `reuse_run_contract` 比较模型、数据和配方；源码及依赖版本摘要作为审计记录保留，不再作为运行门槛。
- `runtime.hf_full_state.HFCheckpointProtocol` / `HFTrainerCheckpointProtocol` 统一 ACT、Ego、π、GR 的完整点发布、发现与轮换；Ψ、λ保留原生断点集成。
- `runtime.numerics.require_finite` 和 `runtime.processes.wait_worker` 复用数值异常传播与子进程回收。ACT/π/GR/Ego 复用 `epoch_safe_training_loop`，Ego 再加 `tail_safe_training_loop` 处理恢复后的 epoch 尾部累计。

剩余结构问题按影响排序：

1. **方法名不能标识实际架构版本。** 当前 Ψ0 `VGQ-VLM` 与 PSI0-RAW 后续同名方法架构不同，见下表。应先记录明确的方法版本与权重结构，避免按同名直接迁移实验或 checkpoint。
2. **扩展边界仍有穿插。** GR00T Query 前向位于原生模型目录；Ψ0 原生模型、Trainer 和数据层有接入钩子。只看 `rhsci/model.py` 无法覆盖全部研究逻辑。应列出原生接入点，并让新增研究实现尽量由明确接口调用。
3. **守护/启动流程结构重复。** 多个 `train` / `launch` / `guardian` 各自处理后台日志、锁、参数转发和 worker 重启。公共完整断点协议已共享，进程生命周期尚未统一；下一步可抽取它，保留各模型的 worker 命令及配方构建。
4. **推理服务接口分散。** π/GR 使用公共 HTTP 层，ACT 独立 FastAPI，Ego 使用 `BaseHTTPRequestHandler`，Ψ/λ保留各自服务。可复用请求编解码、错误响应和 reset 协议，采样本身留在模型侧。
5. **公共缓存构建反向依赖 ACT。** `scripts/data/build_raw_input_cache.py` 从 `act.rhsci.data` 和 `act.rhsci.train` 导入 Dataset/参数工具。通用原始数据读取应归入公共数据层，避免其他骨干构建缓存时依赖 ACT 的训练入口。
6. **推理资产仍与训练集耦合。** π `serve.Server.__init__` 核对整个训练集的内容指纹，Processor 从数据目录加载统计；尚非仅凭模型＋独立处理器资产就能部署。应导出独立统计/处理器资产并兼容旧点。
7. **恢复实现有依赖版本耦合。** Ego `tail_safe_training_loop` 通过读取 HF 源码并替换一行同步条件修复尾部恢复；匹配失败会报错。它有回归覆盖，但维护时仍需检查对应 HF 内部实现，不能等同于稳定公共 API。

旧 `rsci_*` 权重键、配置名和产物名用于读取已有模型，不是另一套重复源码；本轮不因命名清理破坏兼容。

### 是否严格复现

严格复现需要同时说明参照版本、模型结构、初始化权重、数据与预处理、损失、采样、优化器/调度/有效 batch、恢复和部署协议。保留官方模块或若干超参数，只能证明这些部分的来源或对齐；本轮 CPU 回归不是官方全模型等价测试，也不能证明最终成功率相同。

| 对象 | 已确认差异或验证边界 | 当前能作的结论 |
| --- | --- | --- |
| SIMPLE | 固定官方 `b75c6786` 的源码/子模块已核验；level-2 两份公开 ZIP 实际文件相同，不能由此证明 PR 所述物体修正已实施或从未实施 | 源码固定版本对齐；[评测数据修正仍未核验](SIMPLE_OFFICIAL_20261030.md#eval-layout-audit-20261101)，未做物理仿真闭环 |
| Ψ0 最新研究实现 | 当前 `_rhsci_visual_memory` 逐帧取 VLM 输出，默认 Reader 后 `_rhsci_query_latent` 再进语言 Transformer；生产 `RSCI/psi0_controls/methods.py::policy_class` 对 VGQ-VLM 不建旧 Reader，而由 `vlm_query.four_frame_inputs` / `query_output` 将四原图与 Query 直接送入 VLM。关闭当前 Reader 也不会变成后者 | **不是最新 PSI0-RAW 同名方法的严格复现**；后续 State-Reconstruction 等方法也不在当前通用入口中。不能按方法名认定权重兼容 |
| GR00T SIMPLE | 固定官方 [finetune_simple.yaml](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/baselines/gr00t-n1.6/presets/train/finetune_simple.yaml) 指向 `pretrain_he_g1_h1_mixed_scratch_gr00t/checkpoint-50000`，本仓默认 NVIDIA N1.6-3B；另有 map-style 完整动作窗口采样、Query、ZeRO-1 FP32 Adam 和 DiT 重算适配 | 初始化明确不同；优化超参数对齐不等于实验严格复现。已纠正 GR00T README 原“初始化相同”的表格 |
| π0.5 | 复用原生底层模型，但 `encode` / `expert` / `forward` / `sample_actions` 是 RHSCI 自己组织的路径；safe 归一化对退化范围的处理不同；HTTP 用当前实测 STATE 并返回前 24 步，不复刻原生 WebSocket 的预测尾部状态回填 | 当前 RHSCI 适配；尚无完整官方前向/梯度/部署逐项等价验收，不能称严格原版 |
| ACT | 使用当前 SIMPLE 脚本的 16 步配置与原生动作损失，加 EGQ-AE 辅助目标。历史公开 ACT checkpoint 的 100 步配置不是本配置 | 可说明与指定 16 步配方对齐；不能泛称与所有已发布 ACT 权重或实验相同 |
| EgoVLA | 复用 VILA/动作 decoder，但显式组织编码与无 proprio 解码；添加辅助支路与恢复尾部适配。115 epoch 配方需显式 `--preset simple-115epoch`，通用默认不是该配方 | 原生模块上的 RHSCI 扩展；需分别核对预设、数据与执行，尚不能证明全模型等价 |
| λ₀ | 保留父类动作前向和原生 RTC；增加 Reader/Query 与恢复后端；`prepare.safe_bundle` 对变化维度但退化 quantile 范围回退到观测 min/max | 明确存在预处理扩展；当前无配套环境的全模型验证，不能判定严格复现通过 |

“与官方 baseline 一致”和“实现 RHSCI 研究方法”是两个需要分别验收的目标。辅助损失是设计变更，本身不构成实现错误；但必须标明参照方法和版本。本轮未擅自把较新生产方法迁入，也未以文档更正替代算法修复。

### 本轮实际验证

使用本机已有环境，执行 `tools/test_cpu.py --backbone <name>`，离线 CPU 模式，不安装依赖。Ψ0 43、π0.5 37、GR00T 32、ACT 29 项通过；这些套件按 helper 规则排除分布式用例。EgoVLA 29 项中 28 通过，1 项因缺少 `flash_attn` 报错。λ₀没有找到配套环境，在现有 π 环境（Transformers 4.53.2）执行时，3 项轻量测试通过、3 个模型测试模块因缺 `Qwen3_5Config` 导入失败；该骨干锁文件要求 Transformers 5.14.1，不能把不匹配环境的失败判成模型缺陷。

本轮未做六骨干完整官方权重的前向/梯度对照、GPU 正式训练、生产大断点恢复或 SIMPLE 物理闭环。此前生产实验的通过记录不能直接转移为 RHSCI-Private 当前代码的验收。

<a id="simple-shared-interfaces-20261102"></a>

## 参考 SIMPLE 的接口整理（2026-11-02）

本节记录用户确认后的实际整理，覆盖上方审查第 3、4、5 项的旧现状。参照固定官方 SIMPLE `b75c6786` 的 `EnvRunner._resolve_policy`、`PolicyLike`、`HttpActionClient` 和 `SonicActionPolicy`：公共调用层依赖小接口，骨干适配器提供差异。SIMPLE 自身也有不同协议与重复入口，本仓没有将保留原生协议视为必须消除的问题。

| 公共实现 | 已接入的调用方 | 模型侧继续负责 |
| --- | --- | --- |
| `runtime/guardian.py::run_checkpoint_guardian` | π0.5、GR00T 的 `rhsci.guardian.run_training_guardian` | 配方验证、worker 参数、原生完整检查点校验；保留四次尝试、无新点不重复回放、不从零自动重启 |
| `runtime/processes.py::run_worker_process` / `wait_worker` | ACT、Ego 直接调用；π/GR 经公共 guardian 调用 | ACT/Ego 原有的三次尝试和进展判定仍在各自 guardian；公共层负责独立进程组、信号转发及子进程回收 |
| `inference/protocol.py` 的 `RequestMessage`、`ResponseMessage`、`numpy_serialize`、`numpy_deserialize` | ACT、π直接使用；Ego 复用数组编解码；原生 Ψ/GR 等通过旧 `psi.deploy.helpers` 路径重导出同一实现 | 各模型的观测字段、维度、归一化与动作含义；旧字段名和 NumPy wire format 保持 |
| `inference/http_app.py::create_policy_app` | ACT，以及 π/GR 的公共 HTTP 入口 | `predict_action(payload)`、health 内容、可选 reset；ACT 保留 400 错误，π/GR 保留 422；不为无 reset 的策略新增路由 |
| `data/lerobot.py::load_raw_lerobot_dataset` | `scripts/data/build_raw_input_cache.py` | 原生 LeRobot 帧表/视频路径；构建器不再导入 ACT Dataset、Processor 或训练参数，不需要 `stats_psi0.json` |

π worker 参数转发改用既有 `worker_arguments`，与 GR 一致，避免将旧 `--resume=...` 和 guardian 控制参数再次带入恢复命令。分布式进程数、端口计算、缓存选择、NCCL 环境和训练配置保持。公共缓存继续使用既有 schema `act-raw-input-cache-v1`，没有更名旧产物或改写已有缓存。

原生模型接入点明确如下；这些前向不在本轮移动，以保持模型参数键和研究实现边界：

| 原生接入点 | 调用扩展的位置 |
| --- | --- |
| `psi/models/psi0.py::Psi0Model` | 继承 `RHSCIPolicyMixin`；构造/恢复调用 `_init_rhsci` / `_restore_rhsci`，前向调用 `_rhsci_forward`，推理调用 `_rhsci_inference_features` |
| `psi/trainers/finetune.py::FinetuneTrainer` | 训练传入历史及辅助开关并组合 ARDY loss；验证经 `_rhsci_inference_features` 处理条件 |
| `psi/data/lerobot/lerobot_ext.py::LeRobotDatasetWrapper` | 历史采样、ARDY sidecar 与输入缓存接入 |
| `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` | `Gr00tN1d6ActionHead` 构建 `VisualStateQuery`；`Gr00tN1d6.forward` 编码历史、调用共享 DiT Query 并组合辅助损失 |

仍保留模型专属 Processor/Reader、动作损失、优化器配方、采样以及原生服务。Ego 的标准库 HTTP、Ψ/λ的原生服务与 WebSocket 入口继续保留；没有新增重型公共 Trainer。π推理仍依赖训练目录统计，Ego HF 恢复尾部仍耦合内部版本；这两项尚未改变。上节列出的 Ψ方法架构和 GR 初始化差异同样未因基础设施整理而解决，不能据此宣称严格复现。

验证：Ψ套件新增11项后共54项通过（原43项也通过）；π37、GR32、ACT29 项原有 CPU 套件通过。Ego28通过，1项仍缺 FlashAttention。公共守护与真实四骨干子进程终止共7项、原有 HTTP 11项、ACT HTTP 1项、官方 SIMPLE wire format 2项、共享接口3项、原生 LeRobot 数据/无骨干导入2项通过。计数在套件之间有重叠，不是独立测试总数。真实子进程测试覆盖终止转发/回收、失败后完整点恢复、无点不重启、无进展退出、控制参数过滤和路径含空格；数据测试用临时三帧原生 LeRobot 数据，未构建生产全量缓存。HTTP 覆盖原生消息重导出、非连续数组、ACT reset/错误状态、Ego 实际 HTTP 往返及固定官方客户端兼容。原生 Ψ/ACT/DP/GR 部署模块导入、11项公共 CLI 测试、Python 3.10 语法和现有 Ruff 检查通过。仅 CPU；未正式训练、GPU评测、安装环境或改动 PSI0-RAW 生产状态。

## Development

Maintained RHSCI code, launch tools and tests use Python 3.10 syntax, four-space indentation, double quotes, sorted imports and a 100-character target line length. `ruff.toml` scopes these checks; native source retains upstream style.

```bash
uvx --from ruff==0.17.2 ruff format --config ruff.toml --check .
uvx --from ruff==0.17.2 ruff check --config ruff.toml .
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_training_cli.py
```

With the corresponding backbone environment active:

```bash
python tools/test_cpu.py --backbone psi0
```

Replace `psi0` with `pi05`, `gr00t`, `egovla`, `lambda0` or `act`. The helper keeps temporary caches outside the checkout and excludes distributed tests. Optional λ₀ asset tests use `RHSCI_TEST_LAMBDA0_STATS` and `RHSCI_TEST_LAMBDA0_BUNDLE`; missing assets produce skips. CPU tests do not exercise full-model CUDA training or simulator rollouts.

For λ₀ asset tests, `RHSCI_TEST_LAMBDA0_BUNDLE` points to the prepared bundle's `train` metadata directory, not its root (for example `/bundles/my_task-safe/train`). `RHSCI_TEST_LAMBDA0_STATS` points to its normalization JSON. Set `PSI_RUNTIME_ROOT` to the external runtime root when metadata uses paths relative to its `data` directory.

The HTTP checkpoint and protocol tests run in the π0.5 environment with FastAPI and its test client installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_http_inference.py
```


## 2026-10-30 PSI0-RAW integration

Latest input-cache and training-recovery changes are documented in the [integration record](PSI0_RAW_INTEGRATION_20261030.md). It lists source paths, compatibility limits and CPU validation. Existing inference entry points and backbone recipes remain documented above.


## Training responsibilities (2026-10-30)

ACT, π0.5, GR00T and EgoVLA use `config`, `preflight`, `contract`, `trainer`, `worker` and guardian modules. `train.py` remains the compatible CLI entry and reexports existing helpers. See the [responsibility map and validation](TRAINING_STRUCTURE_20261030.md). Raw input caches use explicit `InputCacheConfig`; environment defaults are read only by the CLI.

## π0.5 完整恢复修复（2026-11-02）

[完整恢复修复](PI05_CORRECTNESS_20261102.md)按用户最新要求仅保留第二点：`create_trainer_class` 接入公共 `epoch_safe_training_loop`，π/GR 共用 `isolate_worker_rng`。原归一化、v4 训练契约、模型配置、部署反变换及命令行恢复为本轮修改前的版本，不引入 v3 或 min/max 回退。完整标记不能代替样本进度验证；实际 Trainer CPU 对照覆盖单次、多次与 epoch 边界恢复，八卡长跑/噪声 worker 逐位等价未验证。
