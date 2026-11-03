# Ψ0 方法清理与原仓库处理核对

2026-11-03，所属 RHSCI-Private。用户确认“早期 VGQ-AE”实际指早期 **VGQ-VLM**：删除旧无星号 VGQ-VLM，将后来的四原图直接 VLM Query 恢复为 `VGQ-VLM`；删除状态教师蒸馏，保留状态重建。沿用现有包布局和训练/推理入口。

## 当前方法和函数

| 方法 | 实际调用 | 动作与部署 |
| --- | --- | --- |
| `VGQ-VLM` | `_init_rhsci` 不建 Reader；`_rhsci_forward` → `vlm_query.four_frame_inputs` → `query_output`。四原图 patch、一次当前语言提示、一枚 learned `body_query` 同次进入完整 VLM，沿用原生 mRoPE；`body_query_head` 输出128维 ARDY | 动作分支独立使用当前图像/语言；辅助梯度更新共享 VLM LoRA；推理跳过历史和 Query |
| `STATE-RECONSTRUCTION` | `_init_rhsci` 复用 VGQ-AE 的 Reader、单 Query 和共享 AE blocks，只将辅助头末层128→31；`supervision.state_labels` 提取归一化的当前31维关节标签；`controls.reconstruction_loss` 按有效关节元素归一化 | 真实状态仅为标签；`DataTransform` 最终清零状态输入，动作头也禁用状态条件；部署不需要状态标签 |
| `VGQ-AE`、`EGQ-AE`、`EGQ-VLM` 与带星号方法 | 原有独立方法继续使用其 Reader/Query 路径 | 带星号方法保留将 Query latent 加入动作条件的语义，不将其悄悄替换成新的无星号方法 |

`controls.forward_loss` 集成动作 flow/RTC 与辅助监督，ARDY MSE 继续复用公共 `rhsci.runtime.losses.masked_auxiliary_loss`。状态重建沿用原组的 `ardy_valid` 有效时间窗口，但不使用 ARDY 数值作重建目标。配置、监督提取、模型、损失和 collator 均已移除状态教师蒸馏的入口/参数/标签；ARDY 标签生成所用的视觉教师不属于此次删除对象。

新的 CLI 只接受 `VGQ-VLM`，没有 `VGQ-VLM-DIRECT`。`config.load_run_config` 仅将已有直接四图版本保存的 `VGQ-VLM-DIRECT` 名称映射为 `VGQ-VLM`，其 `body_query` 权重键保持不变。早期同名 Reader 权重会明确报结构不兼容，不能直接续训或推理；没有将旧权重伪转换成新版。文件和历史训练产物未作清理。

## 原仓库处理是否保留

对照本机 PSI0-RAW `9ceb57802f1dd8c0326c5ffed170c17e1776025d` 及当前实际文件 SHA。Ψ0 专门核对 `RSCI/psi0_controls/native` 和 `RSCI/psi0_state_reconstruction`，避免误用更早根目录实现；文件 SHA 与20组数值配对结果见[核对清单](PSI0_PREPROCESSING_AUDIT_20261103.json)。对照是我们原来的 PSI0-RAW；不是宣称覆盖官方新版本的全部可选功能。

| 处理 | Ψ0 整理版当前行为 | 核对结果 |
| --- | --- | --- |
| RGB 缩放/裁剪 | 默认先 Resize 至180×320，再中心裁剪180×320；Resize 的 NEAREST 插值保持原实现；随后交给原生 Qwen processor | 与原专用快照一致；非均匀图像配对逐像素通过 |
| 颜色增强 | 默认训练 `img_aug=True`；brightness 0.2、contrast/saturation (0.8,1.2)、hue 0.05 | 默认入口启用；当前与四帧历史共享一次颜色参数；配对像素及调用后 Torch RNG 一致 |
| EGQ 图像 | 当前 VLM 图像按原尺寸处理，DINO 历史保留原 RGB 全视野与 uint8；颜色参数共享 | 保留，测试覆盖原始历史不被原地修改 |
| 验证/推理 | `no_aug=True`，同一图像几何及处理器，关闭随机颜色增强 | 保留；启用/关闭增强分别测试 |
| 历史采样 | 50 Hz，`[-6,-4,-2,0]`；episode 起点重复边界帧并携带 validity；不跨 episode；辅助损失屏蔽不完整历史 | 时间戳、重打包、padding 和 mask 配对通过 |
| 动作/状态 | SIMPLE 原字段顺序；动作36维、状态32补至36；沿用 `stats_psi0.json` min/max、裁剪及退化范围公式 | 正常/恒定维、bounds/quantile 可选分支及反归一化数值配对通过；默认仍是 min/max |
| 指令/多模态输入 | 指令小写；原生聊天模板、视觉占位、processor patch 与 mask；四图 Query 只保留一次当前提示 | `build_qwenvl_inputs` 函数 AST 一致；新增小型真实 Qwen 前向验证四图 patch 排序与 mRoPE 路径 |
| 动作训练 | 原生 flow matching、RTC 前缀、动作 padding mask 和时间求和损失尺度 | 新对照沿用原专门实现；不改变动作维度或部署去噪入口 |

没有漏掉原 Ψ0 RGB 训练的颜色增强。`GaussianNoise`、椒盐噪声类虽然存在，原对应 RGB 配方并未调用，整理版也不擅自启用。早期 depth/v5 独立研究分支此前就未进入整理版默认 RGB 框架，本轮不将这些旧实验重新混入。

其他骨干同时核对：

| 骨干 | 原有处理与整理版证据 |
| --- | --- |
| π0.5 | `augment_training_images` 函数及底层 `preprocessing_pytorch.py` AST 与原仓库相同：224 resize/pad、原生随机裁剪/颜色变换、四帧共享参数，推理关闭增强。归一化默认仍为原 v2；另保留已有旧 v1 加载选项和显式状态噪声开关，未因本轮清理改公式 |
| ACT | `Processor.__init__`、`__call__`、`inference` 和原生 `ACTModelTransform.__call__` AST 一致：Resize/CenterCrop、ColorJitter、ImageNet Normalize；当前与历史重放同一随机状态，推理关闭增强 |
| GR00T | 原生 `processing_gr00t_n1d6.py` / `image_augmentations.py` AST 一致；`build_processor` 继续启用 albumentations，颜色参数0.3/0.4/0.5/0.08，与原 adapter 相同。继承原生 crop/rotation/resize、同组 replay、动作/状态 processor；缓存只存原始 RGB，样本内执行增强 |
| EgoVLA | 继续原生 `process_image_ndarray_v2`，RGB 明确不翻 BGR，resize 和原生图像标准化；`with_aug=False` 与原 SIMPLE 配方相同，本来就没有开启 Ψ0 式随机颜色增强。prompt 中仅调整 worker 模板恢复位置，图像/动作处理保持 |
| λ₀ | 仍经原生 PsiSimpleHandler/processor；现整理版另有已记录的 safe bundle 归一化适配。当前原树未找到对应的 RSCI λ adapter，不能把这一部分说成逐函数完全相同；本轮无该模型全环境验证 |

复核可使用原 Ψ0 依赖环境，显式给出原仓库和报告路径：

```bash
PYTHONDONTWRITEBYTECODE=1 "$PSI0_PYTHON" -B tools/audit_psi0_preprocessing.py \
  --source-root /absolute/path/to/PSI0-RAW --report /tmp/psi0-preprocessing-audit.json
```

该工具先核对增强类及默认参数 AST，再做数值配对，避免参照模块导入整理版增强类时掩盖差异；不要求公开克隆用户保留原仓库即可正常训练。

## 验证与附带修复

- Ψ0 20组原/整理版配对检查，逐像素/数组及相关 RNG 状态一致；清单记录实际源文件 SHA，未读取活动大权重。
- Ψ0 CPU 回归覆盖新版真实小型 Qwen 四图前向、视觉 LoRA 与 Query 梯度、单帧动作隔离、严格权重往返、旧同名结构拒绝、状态重建监督隔离及增强。其他骨干使用各自已有环境运行 CPU 回归。结果：Ψ0 59、π0.5 39、GR00T 34、ACT 30、EgoVLA 29，共191项通过；分布式项按现有 CPU runner 排除。
- 回归发现 ACT 的 HF 4.57 已从恢复位置初始化 `step`，公共尾恢复不再对它引用旧 `steps_skipped`；HF 4.51 仍保留原修正。ACT 原零/多 worker、累计尾对照通过。
- 本机文件时间戳粒度可让快速同大小覆盖后的 stat 不变，Ψ0 数值缓存会沿用旧坏结果；去掉该缓存，保存/恢复直接读取实际张量校验。未增设启动门禁。

CPU 对照不代表生产尺寸 GPU、完整旧模型迁移或 SIMPLE 物理闭环通过。只修改整理版源码和文档，不操作原仓库活动训练/断点，不推送远端。
