# PSI0-RAW 训练、恢复与推理代码整理

2026-10-31 目录迁移提示：本文保留该日期提交的历史路径与验证记录。当前模型专属代码已移入各原生包，使用[现行位置映射](NATIVE_PACKAGE_LAYOUT_20261031.md)；旧模块路径不可再作为执行入口。

日期：2026-10-30。目标：RHSCI-Private / main，原 HEAD `0c4656e834d4cde8ccb22ac135508b5f07466f24`。
来源：本机 PSI0-RAW / main，HEAD `9ceb57802f1dd8c0326c5ffed170c17e1776025d`，**包含未提交代码**；因此 HEAD 不能单独标识本次来源，逐文件 SHA 与目标映射见[来源清单](PSI0_RAW_INTEGRATION_20261030.json)。

此文记录首次整合提交 `7c39b9ac9b0285f64596b0ea00e96210dd2009ba` 的历史状态；随后模块拆分、显式缓存 API 和更新验证见[训练结构整理](TRAINING_STRUCTURE_20261030.md)。下方旧测试环境限制不代替后续验收。

## 代码去向与取舍

| 来源 | 整理位置 | 处理 |
| --- | --- | --- |
| `src/rsci_pi05` | `src/rsci/backbones/pi05` | 接入原始输入缓存；修正 STATE/NO-STATE 保存配置中的历史/辅助标识。保留目标已有状态噪声、通用路径和 reference/history/query/both 执行模式 |
| `src/gr00t/gr00t/rsci` | `src/rsci/backbones/gr00t` | 接入原始缓存、epoch/index 采样、逐样本增强 RNG 与恢复后 sampler epoch；独立 loader generator 避免重建 loader 消耗模型 RNG。保留目标 Zero1 优化器和 DiT 重算 |
| `RSCI/egovla/rsci_egovla` | `src/rsci/backbones/egovla` | 接入原始缓存、局部 HF epoch 尾部恢复、worker 对话模板恢复、真实 spawn 预检与实际 VLM 执行核验 |
| `RSCI/act/rsci_act/space_cache.py` | `src/rsci/data/raw_input_cache.py` | 共享只读表和原始 RGB/PTS mmap；不再通过 ACT 私有模块给其他骨干提供依赖 |
| `RSCI/pi05/input_cache.py` | `src/rsci/data/input_cache.py` | π0.5/EgoVLA/GR00T 复用同一帧和 episode 表读取实现，保留各自环境变量和配方 |
| `RSCI/act/rsci_act/video_cache.py` | `src/rsci/backbones/act/video_cache.py` | 接入共享 mmap，未启用缓存时保留原生视频解码路径；保留目标已有 CPU RNG 隔离 |
| `RSCI/act/tools/build_act_space_cache.py` | `scripts/data/build_raw_input_cache.py` | 显式 dataset/cache-root；保留原视频、锁、独立随机 seek 验证、逐文件 SHA 与原子发布；不再硬编码任务或机器路径 |
| ACT/EgoVLA/GR00T 恢复与缓存测试 | `tests/act`、`tests/egovla`、`tests/test_gr00t_resume.py` | 迁入恢复/缓存回归；补共享缓存复制/认证与 worker 模板重建测试 |

对照过 `src/psi`、`src/openpi/models_pytorch`、GR00T 原生模型和 EgoVLA 视觉入口。目标已有研究/通用接口不能用来源普通版本整体覆盖；π0.5 原生执行修改已经包含在目标。各骨干推理仍走 [Inference](INFERENCE.md) 中的原入口；EgoVLA 训练/服务共用修正后的 Processor。没有新增推理协议或改动作坐标、归一化、RTC。

来源的按日期训练控制器、现场 PID/tag、接管和上传脚本留在 PSI0-RAW：它们绑定原仓库源码 SHA、路径和活动契约，直接搬迁会使旧训练恢复失效。五任务使用目标现有显式 `--task/--dataset/--sidecar`，无需复制机器专属任务别名表。λ₀ 等目标独有实现保留。

## 使用和兼容性

执行位置为 RHSCI-Private 根目录，各骨干环境/权重/参数见 [Training](TRAINING.md) 及 baselines 指南。缓存是可选项，默认关闭；需要时将下列变量设为包含 `<dataset目录名>/manifest.json` 的**缓存父目录**：

| 骨干 | 环境变量 |
| --- | --- |
| ACT | `RSCI_ACT_SPACE_CACHE` |
| π0.5 / EgoVLA | `RSCI_INPUT_CACHE` |
| GR00T | `RSCI_GR00T_INPUT_CACHE` |

格式为 `act-raw-input-cache-v1`，仅适用原生 50Hz、3×360×640 uint8 RGB，含完整原始小张量表和 PTS；数据 fingerprint、清单 SHA 与源视频 stat 必须相符。Guardian 对缓存和视频完整 SHA 验证，分布式 worker 保留清单/stat 门禁。缓存身份与共享实现均纳入训练契约，不缓存增强/归一化/模型表征。

缓存生成器提供 `--dataset`、`--cache-root` 和 `--workers`（默认4）；`--help` 已检查，**本轮未执行生成**。输出必须位于代码目录外，准备数据及使用生成器前核实空间与源数据版本。

EgoVLA epoch 预算改为 `epochs × ceil(loader_microbatches / accumulation)`，保留每个 epoch 尾部不足累计组的更新。局部修复 HF4.53.2 的恢复同步判断，未更改 ACT/GR00T 的共享循环。`--gradient-checkpointing/--no-gradient-checkpointing` 可显式控制视觉/语言重算；目标默认仍为启用，115epoch/16×8×4/full VLM 与原生优化器不变。其他 HF 循环不匹配时明确拒绝启动。

这是源码整合，**不是旧运行/断点迁移**。来源和目标的文件路径、源码 SHA、缓存和 sampler 契约不同；原断点不会自动获得新契约。不要重写旧 launch_contract 或跳过校验，新运行需独立输出目录；旧断点推理需使用与其匹配的代码和环境。

## 验证范围

使用已有独立 Python 环境，仅运行 CPU 微型模型和临时 fixture，没有下载、安装、训练正式模型、读取活动 checkpoint 或运行 GPU/仿真评测。

- π0.5：36 项通过，包括执行模式、前向/梯度、归一化/增强、STATE 噪声和推理回归。
- GR00T：31 项通过，包括微型模型、优化器与中途恢复；新增恢复测试核对 dropout 下连续/中断训练的消费序列和最终参数。
- EgoVLA：27 项通过，包括原生动作专家、完整 Adam/RNG 恢复及新增 epoch 尾部恢复；另 1 项 worker 模板序列化回归通过。
- ACT：24 项通过（含原有8项缓存测试）；共享缓存新增2项单独通过，确认增强不能写回 mmap、Guardian 拒绝同大小篡改。
- 通用训练入口：11 项通过。Python 语法、Git diff 空白和源码清单核验结果见最终收据。

本沙箱禁止测试所需 socket，排除 DDP、持久 worker 和部分服务测试；初次运行的多进程测试因 IPC 被拒绝而中止，只停止了本轮测试进程。EgoVLA SigLIP 的完整导入测试受缺失 FlashAttention 阻塞，未假装通过。GR00T 测试进程仅关闭其未使用的可选 DeepSpeed 检测，以绕开沙箱 PID 探测限制；生产检测保持原样。未运行 Ruff（当前未安装且本轮不安装）。完整多进程、全尺寸资产、GPU、吞吐和闭环仍待在对应运行环境验收。

## 资产交接

本次只整理源码和小文件。数据、权重、环境、缓存和运行产物外置；model `ds1x/RHSCI/checkpoint` / dataset `ds1x/RHSCI/train-data`、`eval-data` 的既有分工保持。没有新增资产上传或删除。代码提交 SHA 与推送状态以本次交付为准，不把来源旧提交写成目标新提交。
