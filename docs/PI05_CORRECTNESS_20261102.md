# π0.5 完整恢复修复（2026-11-02）

所属：RHSCI-Private 当前主树；修改或解释 π0.5 恢复逻辑前必读。按用户最新要求“优化第二点就好”，本轮最终只保留样本 epoch 恢复与 worker RNG 隔离。此前提交 245f07c 的 v3 归一化方案已撤回；该范围更正覆盖本文件及同日索引先前的 v3 默认说明。

## 最终保留的修改

| 函数 | 职责 |
| --- | --- |
| `openpi.rhsci.trainer.create_trainer_class` | 给 π Trainer 接回公共 `epoch_safe_training_loop`，与原 PSI0-RAW 已有接入对应 |
| `rhsci.runtime.hf_resume.skip_preserving_epoch` | 跳过已消费 batch 时，给新 loader 恢复真实 epoch，再开始迭代/预取 |
| `rhsci.runtime.hf_resume.isolate_worker_rng` | worker 启动和重建使用独立 generator，不扰动模型动作噪声/dropout 的主 RNG |
| `gr00t.rhsci.trainer.create_resume_trainer_class` | 改用同一个隔离函数，保持 GR 已有行为，避免重复实现 |

Accelerate 的 `skip_first_batches` 会新建 `DataLoaderShard`，新对象 iteration 默认 0；即使原 loader 已设置真实 epoch，新对象首次迭代仍可能把 sampler 重置回 epoch 0。原 PSI0-RAW 的 π Trainer 已有 epoch 修复，整理为 RHSCI 时漏接；此前只测断点完整和能够继续训练，没有覆盖跨 epoch 数据顺序，是迁移验证遗漏。

只补 epoch 后，持久 worker 重建抽取 base seed 仍可能消耗主线程随机数，从而改变模型噪声/dropout。现复用 GR 已有的独立 generator 做法。完整标记和 SHA 继续验证文件完整性，不能代替样本进度测试。

## CPU 对照证据

`tests/test_pi05_resume_progress.py` 使用实际生产 Trainer、CPU AdamW 和完整保存回调；19 样本、每批 2、累计 1、总计 24 步。中断前后的总步数和调度器定义保持相同，比较以下场景：

- 第 13 步完整点恢复，位于第二个 epoch 中途；
- 两个持久 worker，第 13 和 21 步两次恢复；
- 第 20 步 epoch 边界恢复。

修复前样本顺序不一致；只补 epoch 后，持久 worker 的主随机流仍有差异。两项修复后，三种情况的样本序列、前向 Torch/Python/NumPy 随机值、最终模型参数、Adam、调度器及主 RNG 状态逐元素一致。

该对照只覆盖确定性 Dataset、默认累计 1 的 CPU 场景。worker 内随机 STATE 噪声状态不随检查点保存，因此不能承诺噪声开启、其它累计配置、真实八卡、全模型或长程训练逐位等价。当前结果不代表所有骨干严格复现已通过。

## 配方与兼容性

归一化 Processor、训练契约、模型配置、worker 装配、服务端反变换和相关历史测试均恢复至 1c579c3 的内容。继续使用原 `simple-quantile-zero-span-clipped-v2` / `SIMPLE-pi05-default-v4`，没有新增归一化选项，不需要为本次恢复修复更换 run-tag 或迁移旧断点。

归一化稀有变化问题按用户决定本次不改。先前的[数据审计](PI05_NORMALIZATION_AUDIT_20261102.json)只保留为历史证据，包含已撤回原型的往返结果，不能据此认为当前代码已经启用 v3。用户检查后没有启过正式 v3 训练；本次不改动生产训练、数据、模型权重或旧完整点。

撤回归一化后的验证：π CPU 38 项、GR CPU 32 项、HTTP 11 项及 CLI 11 项通过。仅本地提交，不推送远端。
