# π0.5 稀有变化与完整恢复修复（2026-11-02）

所属：RHSCI-Private 当前主树；检查训练归一化、恢复和部署兼容性时必读。本记录更新此前只验证数值有限、检查点完整或能继续训练的结论；那些检查不能证明训练信息与数据进度没有丢失。

## 与 PSI0-RAW 的实际区别

对照的是 PSI0-RAW 当前工作树，不能将其当作干净的已提交快照。源文件 SHA 和真实数据核验见[审计记录](PI05_NORMALIZATION_AUDIT_20261102.json)。

| 问题 | PSI0-RAW 当前实现 | 修复前的 RHSCI-Private |
| --- | --- | --- |
| 退化分位数抹去稀有动作/STATE | `src/rsci_pi05/processing.py::Processor.normalize` 同样按 q01≈q99 直接置零，问题也存在 | 沿用旧实现，并非原仓库已解决的迁移差异 |
| 第二个 epoch 中途恢复后 shuffle 顺序改变 | `src/rsci_pi05/train.py::trainer_class` 已接入 `epoch_safe_training_loop(Trainer)` | `openpi.rhsci.trainer.create_trainer_class` 漏接；其他骨干已使用公共实现 |

2026-10-22 的 QSAFE 修复避免退化分母导致百万级标签及 NaN，但“输出有限”不等于“保留真实变化”。旧记录中的数值通过不能证明稀有动作信息没有丢失。此次未修改 PSI0-RAW 生产源码、训练进程、环境、配方或断点。

## 归一化、保存与部署

新训练默认 `simple-quantile-rare-range-clipped-v3`，配方版本为 `SIMPLE-pi05-default-v5`。

1. `normalization.build_normalization_ranges(stats)` 为 action、states 构建逐维有效范围：正常 q01/q99 跨度继续使用，退化跨度回退真实 min/max。
2. 只有有效 min=max 才视为恒定并置零；分位数相等而 min<max 的维度保留变化。有效范围线性映射至 [-1,1]；真正恒定维度反变换为原常量。v3 不再给非零分母额外加 1e-6，正反变换共用同一跨度；正常分位数范围外仍会裁剪。
3. `Processor.normalize` 处理动作监督和 STATE 文本输入；`Processor.denormalize` 处理部署反变换，服务端直接调用，避免再写一套公式。
4. `contract.build_training_contract` 将版本及范围纳入训练身份；`worker.run_training_worker` 把处理器的同一版本、范围写入 `rsci_config.json`。服务加载保存范围并核对训练契约，不另算一套。
5. 退化维度缺少真实 min/max 时明确报错。只读审计已核两任务全部 Parquet action/states 的实际极值与元数据匹配；没有读取图像/权重或改写统计。

烤箱 37,903 帧、99 条轨迹：action 高度 q01=q99≈0.74，实际范围约 [0.42,0.74]；旧实现抹去 376 帧动作变化、375 帧 STATE 高度变化，均在 episode_000000。两桌 58,637 帧、91 条轨迹：左手七维受影响，六维为 52 帧，第一维严格非零计 53 帧，其中 52 帧变化超过 1e-6，均在 episode_000030。这是全量 Parquet 行数，不是 30 步窗口数；不同计数口径应分别说明。

上述维度 v3 归一化往返最大误差：烤箱 0，两桌不超过 1.4901161193847656e-8。HTTP 预测路径验证高度 0.42/0.74 分别还原；STATE 高度 token 分别为 0/255。这不代表整个动作空间没有分位数裁剪，也不代表闭环效果已通过。

## 完整恢复

`create_trainer_class` 重新绑定公共 `runtime.hf_resume.epoch_safe_training_loop`。Accelerate 的 `skip_first_batches` 会新建 `DataLoaderShard`，新对象 iteration 默认 0；即使原 loader 已设置真实 epoch，新对象首次迭代仍可能将 sampler 重置回 epoch 0。公共 `skip_preserving_epoch` 在迭代/预取前恢复真实 epoch。

持久 worker 重建抽取 base seed，若使用全局 Torch RNG，会扰动模型动作噪声/dropout。将 GR00T 已有独立 generator 做法提取为 `isolate_worker_rng`，π/GR 共用，GR 行为不变。完整标记和 SHA 验证文件完整性，不能代替数据进度测试。

实际生产 Trainer、CPU AdamW 和完整保存回调的回归测试使用 19 样本、每批 2、累计 1、总计 24 步，保持中断前后总步数/调度器定义相同，比较：

- 第 13 步中断后恢复，位于第二个 epoch 中途；
- 两个持久 worker，第 13 和 21 步两次恢复；
- 第 20 步 epoch 边界恢复。

修复前第一种样本序列不一致；只补 epoch 后，持久 worker 主随机流仍不一致；两项修复后，三种情况的样本序列、前向 Torch/Python/NumPy 随机值、最终参数、Adam、调度器及主 RNG 状态逐元素一致。这里只覆盖确定性 Dataset、默认累计 1 的 CPU 对照；worker 内随机 STATE 噪声状态不随检查点保存，不能承诺噪声开启、其它累计配置、真实八卡、全模型或长程训练逐位等价。

## 历史断点与继续训练

v1/v2 推理保留历史公式，不将旧权重静默解释为 v3。新范围改变训练标签和 STATE 表示，旧训练已经丢失的信息无法只靠替换推理端恢复。采用 v3 应建立新 run-tag 并重新训练；若以后选择旧权重初始化新实验，应明确记录为配方迁移，不能冒充原运行完整续训。本次没有启动这些训练。

若明确需要沿用历史 v2 训练语义继续已有完整点，可给原训练命令增加：

```bash
--normalization simple-quantile-zero-span-clipped-v2
```

这保留 v4 契约身份和旧归一化缺陷，同时使用修复后的样本 epoch 恢复逻辑。默认 v3 与已有 v2 输出目录不兼容，会要求新 run-tag；不自动迁移历史完整点或重置优化器。

验证：π CPU 44 项、GR CPU 32 项、HTTP 11 项通过。新增 π 测试含实际完整恢复、稀有/恒定/近似相等分位数、STATE token、部署往返、保存范围与历史配方兼容。计数有重叠。未正式训练或 GPU 评测；仅本地提交，不推送远端。
