# 守护与恢复问题的原仓库对照（2026-11-02）

所属：RHSCI-Private/main。修改 λ₀ 守护或解释 π0.5/EgoVLA 的精确恢复保证前必读。用户要求“原仓库有吗，没有的修复”，本轮以 PSI0-RAW 当前工作树对照，仅修 RHSCI 独有的 λ₀ 守护问题；π/Ego 两个同源问题按该范围保留。

## 对照结论

| 用户指出的问题 | PSI0-RAW 当前实现与 CPU 对照 | 本轮处理 |
| --- | --- | --- |
| λ₀ 守护退出，worker 留存且训练锁提前释放 | 当前主树没有对应 RHSCI λ₀ 守护；历史提到的 RSCI/lambda0 路径现已不存在。首次整合记录明确将 λ₀ 列为目标独有实现，不能声称原仓库此处已经修复 | 修复 RHSCI 的守护，复用公共进程管理 |
| π0.5 累计>1，中途恢复漏掉 epoch 尾更新 | 原 `src/rsci_pi05/train.py` 只接 `epoch_safe_training_loop`，未接尾部修复。直接运行原 Trainer：19样本/batch2/累计4/3epoch，第4步恢复后8步，连续训练9步；样本和前向随机数一致，参数不同 | 原仓库同样存在，本轮不改；默认累计1不触发该问题 |
| EgoVLA 持久 worker 重建扰动模型 RNG | 原 `RSCI/egovla/rsci_egovla/train.py::get_train_dataloader` 也没有独立 worker generator。直接运行原 Trainer、两个持久 worker：均完成9步，样本一致，前向随机数及最终参数不同 | 原仓库同样存在，本轮不改 |

[机器可读证据](RECOVERY_SOURCE_AUDIT_20261102.json)记录原文件 SHA、实际入口、依赖版本及微型对照结果。原主树包含未提交代码，不能只用 HEAD 代替其源码身份。探针用临时 CPU 小模型/HF 检查点，不读取生产大权重或活动滚动点；未修改原源码、环境、配方、训练进程或断点。微型恢复结果不能量化成功率影响。

## λ₀ 最终调用链

`lambda0.rhsci.launch.run_training_guardian` 持有 guardian.lock → `runtime.processes.run_worker_process` 建立独立 worker 进程组 → `wait_worker` 转发 SIGTERM/SIGINT 并回收直接子进程 → λ₀ 的 finally 按 PID/启动时间清理已观察的同组 rank/子孙 → 全部退出后离开外层持锁块。

- `run_worker_process` 增加可选 `on_start(process)` / `on_wait()`，供 λ₀ 原子写 worker.json、持续转发训练日志和观察子进程；ACT/Ego/π/GR 原调用方式不变。
- 即使启动记录或日志回调异常，公共 finally 也会终止并等待直接 worker；λ₀ 的 finally 继续清理同组子孙，避免异常路径提前释放锁。
- guardian.lock FD 通过 pass_fds 交给 torchrun worker。SIGKILL 无法捕获或转发，但 worker 活着时仍持有锁，同目录新启动会被拒绝。此保护不等于 SIGKILL 能自动结束训练；也不承诺 worker 自行脱离进程组且关闭锁 FD 后仍能覆盖其生命周期。
- 正常完成也先清理残留 rank，再释放锁。重试次数、完整点校验、数值错误不重试、训练命令/配方不变。前台 Ctrl+C 仍只退出 tail；要停止训练应向守护发停止信号。

## 验证范围

修复前真实 CPU 子进程可复现 SIGTERM/SIGINT 后 worker 留存，以及 SIGKILL 后锁可重取。修复后 `tests/lambda0/test_guardian_processes.py` 四项测试通过：SIGTERM/SIGINT 的进程组清理与锁顺序、活动时重复启动拒绝、SIGKILL 时 worker 继承锁、启动记录写失败清理、正常完成残留 rank 清理（部分是同一测试中的子场景）。已有预算完成检查1项通过。

公共进程管理回归由 Ψ CPU 套件54项通过，其中包含 ACT/Ego/π/GR 实际子进程终止与恢复测试；CLI11项通过。没有运行 λ₀ 全模型测试、正式 GPU 训练或评测，也没有安装其缺失的配套环境。π/Ego 本轮只做原 Trainer 的 CPU 对照，未将它们的两个已确认问题写成已修复。

本轮保留用户先前决定：π0.5 归一化不改。仅本地提交，不推送远端。
