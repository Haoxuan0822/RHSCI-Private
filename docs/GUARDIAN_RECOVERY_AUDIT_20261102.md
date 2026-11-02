# 守护与恢复问题的原仓库对照（2026-11-02）

所属：RHSCI-Private/main。修改 λ₀ 守护或解释 π0.5/EgoVLA 的精确恢复保证前必读。2026-11-03 用户更正：“原仓库”指 **SIMPLE 官方**，不是 PSI0-RAW。此前对照对象选错，由此得出的“π/Ego 按用户范围保留”决定已撤回；当前三个问题均已在 RHSCI 修复。下面保留 PSI0-RAW 对照事实，仅供历史来源追溯。

## 2026-11-03：SIMPLE 官方对照及最终处理

只读查询官方远端，SIMPLE main 仍为 [`b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11`](https://github.com/physical-superintelligence-lab/SIMPLE/tree/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11)。本机固定源码433文件及9递归子模块再次核验通过。官方 [README](https://github.com/physical-superintelligence-lab/SIMPLE/blob/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11/README.md) 区分仿真客户端和外部策略服务器；[π客户端](https://github.com/physical-superintelligence-lab/SIMPLE/blob/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11/src/simple/baselines/pi05.py) 调用 WebSocket 推理，[Ego客户端](https://github.com/physical-superintelligence-lab/SIMPLE/blob/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11/src/simple/baselines/egovla.py) 继承 `Psi0Agent`；顶层树没有 λ₀ 路径，也没有这里的 RHSCI Trainer/训练守护。

因此，这三处实现没有可直接对应的 SIMPLE 官方训练函数，不能据此声称“官方训练已证明没有此类问题”，也不能以本地 PSI0-RAW 存在同类问题作为不修复 RHSCI 的理由。补充核对作者另一个公开 Psi0 main `cda3fdf`：[π训练入口](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/baselines/pi05/train_pi05.sh) 调原生 [PyTorch 循环](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/src/openpi/train_pytorch.py)，每批 backward/step，没有这里的 HF 累计尾部判断；这也不等于其恢复具有精确样本/RNG保证。未运行官方全模型恢复实验。

| RHSCI 函数 | 最终修改与证据 |
| --- | --- |
| `lambda0.rhsci.launch.run_training_guardian` | 保留此前信号转发、子孙清理及继承锁修复，细节见下文 |
| `rhsci.runtime.hf_resume.tail_safe_training_loop` → π/Ego `create_trainer_class` | 抽取 Ego 已有修正到公共层，π复用；同步判断计入 `steps_skipped`，恢复后仍更新 epoch 尾部。Ego旧模块只重导出，避免复制算法 |
| `human_plan.rhsci.trainer.RHSCITrainer.get_train_dataloader` | 复用 π/GR 已有 `isolate_worker_rng`；启动持久worker时不再消耗模型主RNG。训练退出复用 `close_dataloaders` 回收已创建loader |

修复前扩展真实 Trainer 测试，π19样本/batch2/累计4/3epoch、第4步恢复，在0与2worker都复现连续9步/恢复8步；Ego已有尾更新测试改为2持久worker后参数不一致。修复后 π 两种worker均9步，样本、前向Torch/Python/NumPy随机值、参数、Adam、调度器和主RNG逐元素一致；原累计1的24步单次/多次/边界恢复也通过。Ego 0与2worker的样本、参数、Adam、调度器及Torch主RNG完全一致。π CPU套件39项通过；Ego独立环境29项CPU中28项通过（含0/2持久worker恢复），另1项因缺少既有 `flash_attn` 依赖报错；GR共享层回归32项通过。

不改变归一化或优化配方；修复不会补回旧训练已经漏掉的更新。独立generator隔离不等于保存worker内部随机变换状态，未验证随机worker增强/噪声、全模型多卡或GPU长程逐位等价。仅本地提交，不推送远端。

## 历史：2026-11-02 PSI0-RAW 对照（不是官方结论）

| 用户指出的问题 | PSI0-RAW 当前实现与 CPU 对照 | 本轮处理 |
| --- | --- | --- |
| λ₀ 守护退出，worker 留存且训练锁提前释放 | 当前主树没有对应 RHSCI λ₀ 守护；历史提到的 RSCI/lambda0 路径现已不存在。首次整合记录明确将 λ₀ 列为目标独有实现，不能声称原仓库此处已经修复 | 修复 RHSCI 的守护，复用公共进程管理 |
| π0.5 累计>1，中途恢复漏掉 epoch 尾更新 | 原 `src/rsci_pi05/train.py` 只接 `epoch_safe_training_loop`，未接尾部修复。直接运行原 Trainer：19样本/batch2/累计4/3epoch，第4步恢复后8步，连续训练9步；样本和前向随机数一致，参数不同 | 当时保留；2026-11-03更正范围后已修RHSCI，未改PSI0-RAW |
| EgoVLA 持久 worker 重建扰动模型 RNG | 原 `RSCI/egovla/rsci_egovla/train.py::get_train_dataloader` 也没有独立 worker generator。直接运行原 Trainer、两个持久 worker：均完成9步，样本一致，前向随机数及最终参数不同 | 当时保留；2026-11-03更正范围后已修RHSCI，未改PSI0-RAW |

[机器可读证据](RECOVERY_SOURCE_AUDIT_20261102.json)记录原文件 SHA、实际入口、依赖版本及微型对照结果。原主树包含未提交代码，不能只用 HEAD 代替其源码身份。探针用临时 CPU 小模型/HF 检查点，不读取生产大权重或活动滚动点；未修改原源码、环境、配方、训练进程或断点。微型恢复结果不能量化成功率影响。

## λ₀ 最终调用链

`lambda0.rhsci.launch.run_training_guardian` 持有 guardian.lock → `runtime.processes.run_worker_process` 建立独立 worker 进程组 → `wait_worker` 转发 SIGTERM/SIGINT 并回收直接子进程 → λ₀ 的 finally 按 PID/启动时间清理已观察的同组 rank/子孙 → 全部退出后离开外层持锁块。

- `run_worker_process` 增加可选 `on_start(process)` / `on_wait()`，供 λ₀ 原子写 worker.json、持续转发训练日志和观察子进程；ACT/Ego/π/GR 原调用方式不变。
- 即使启动记录或日志回调异常，公共 finally 也会终止并等待直接 worker；λ₀ 的 finally 继续清理同组子孙，避免异常路径提前释放锁。
- guardian.lock FD 通过 pass_fds 交给 torchrun worker。SIGKILL 无法捕获或转发，但 worker 活着时仍持有锁，同目录新启动会被拒绝。此保护不等于 SIGKILL 能自动结束训练；也不承诺 worker 自行脱离进程组且关闭锁 FD 后仍能覆盖其生命周期。
- 正常完成也先清理残留 rank，再释放锁。重试次数、完整点校验、数值错误不重试、训练命令/配方不变。前台 Ctrl+C 仍只退出 tail；要停止训练应向守护发停止信号。

## 验证范围

修复前真实 CPU 子进程可复现 SIGTERM/SIGINT 后 worker 留存，以及 SIGKILL 后锁可重取。修复后 `tests/lambda0/test_guardian_processes.py` 四项测试通过：SIGTERM/SIGINT 的进程组清理与锁顺序、活动时重复启动拒绝、SIGKILL 时 worker 继承锁、启动记录写失败清理、正常完成残留 rank 清理（部分是同一测试中的子场景）。已有预算完成检查1项通过。

2026-11-02的公共进程管理回归由 Ψ CPU 套件54项通过，其中包含 ACT/Ego/π/GR 实际子进程终止与恢复测试；CLI11项通过。没有运行 λ₀ 全模型测试、正式 GPU 训练或评测，也没有安装其缺失的配套环境。当时π/Ego仅做PSI0-RAW Trainer CPU对照；2026-11-03实际修复和验证见上节。

本轮保留用户先前决定：π0.5 归一化不改。仅本地提交，不推送远端。
