# 代码检查与可读性复查（2026-11-03）

所属：RHSCI-Private/main，基于 `b671967` 后的当前代码。阅读或修改训练恢复、进程生命周期和模块职责前参考。本轮检查六个 RHSCI 扩展的入口与公共恢复/进程代码；不把原生依赖的全部源码当成已审查，也不重新判定所有模型严格复现。

## 发现和修复

| 问题 | 修复及验证 |
| --- | --- |
| ACT、GR00T 仍只接 epoch 恢复，未包含 π/Ego 已修的累计尾更新 | 在HF4.53.2环境直接运行两者真实Trainer工厂，19样本、batch2、累计4、checkpoint4：连续9步而恢复8步，0/2worker都发生。现四个HF骨干统一调用公共 `tail_safe_training_loop`；ACT也复用 `isolate_worker_rng` |
| GR原生HF4.51.3按floor计算每轮更新数，与实际尾更新不同 | 在GR独立环境继续测试发现：修同步条件后虽都到9步，恢复仍从错误epoch开始，样本62个而连续57个。现公共 `tail_safe_training_values` 复用原始预算、仅在floor与实际ceil不同处修正轮数/步数，GR接入；HF4.53.2已用ceil时原样返回。覆盖显式步数及3epoch两种预算 |
| λ₀ 的 `on_start` 在注册信号处理前执行，停止窗口仍可留存worker | 在真实启动记录回调中暂停后发SIGTERM，旧守护直接退出-15，清理未执行。公共信号作用域现从Popen前延续至退出清理完成，Popen返回前到达的停止信号在取得子进程后补发；λ通过 `on_exit` 在同一作用域清理子孙，再释放锁 |

前者是同类恢复修复遗漏了其他调用方；后者是只测试“等待训练中停止”，漏掉启动/退出回调阶段。完整断点标记、正常终止测试和已有累计可整除的测试都不能替代这些边界验证。没有改变模型结构、优化器配方或归一化；旧训练已漏掉的更新不会自动补回。

修复前GR累计>1且尾部不足一组的epoch预算也可能已经影响调度器总步数；不自动改写这类旧检查点，不能承诺它与修复后重新连续训练的状态相同。恢复一致性测试使用同一份修复后的代码生成和读取检查点。

## 当前函数职责

| 入口/函数 | 输入与职责 |
| --- | --- |
| 各骨干 `train.main` | 解析参数并选择预检、worker或guardian；配置/模型仍由对应模块负责 |
| `config.build_training_arguments` | 用各骨干配方构造原生训练参数，不在公共层统一学习率/损失 |
| `worker.run_training_worker` | 在当前rank装配模型、数据、Trainer和完整恢复回调并执行训练 |
| `trainer.create_trainer_class` | 返回带骨干行为的Trainer子类；GR另有 `create_resume_trainer_class` 作为其分片优化器下层 |
| `hf_resume.tail_safe_training_loop` | 组合真实epoch恢复与累计尾部同步；四个HF骨干共用同一算法 |
| `hf_resume.tail_safe_training_values` | 修正HF4.51按floor计算的更新预算，供GR接入；保留原生样本统计与无长度loader行为 |
| `hf_resume.isolate_worker_rng` | 给loader独立generator；四个HF骨干共用，不负责恢复worker内随机变换状态 |
| `processes.run_worker_process` | 持有信号作用域，创建worker、调用启动/等待/退出回调，并回收直接子进程 |
| `processes._WorkerSignalHandler` | 记录停止信号并向worker进程组转发；信号回调不调用wait，避免重入 |
| `processes._wait_worker` / `_terminate_worker` | 分别负责等待/停止超时升级，以及异常路径的直接子进程清理 |
| λ `launch.record_worker` / `observe_worker` / `cleanup_worker` | 分别记录PID、读取日志并观察子孙，以及按PID/启动时间清理子孙。锁仍由外层guardian持有 |

λ正常结束、SIGTERM/SIGINT和回调异常都先完成清理再退出持锁块。SIGKILL不能捕获，沿用worker继承锁FD防止其存活时重复启动；不声称硬杀会自动终止所有训练。公共 `wait_worker` 保留原调用接口；完整创建/回收逻辑由 `run_worker_process` 提供。

## 可读性判断

**模块主干已清楚，恢复适配和少数入口仍需要较多背景知识。** ACT、π、GR、Ego的配置/预检/契约/worker/Trainer已分开，训练入口没有另建平行模型树；同一恢复算法和进程生命周期现在有明确公共所有者。

本轮删除四个HF扩展中重复的“仅拆分职责”模板注释；保留解释数值/数据/配方原因的注释。πTrainer的 `*a/**kw` 改为标准 `*args/**kwargs`，公共尾恢复用10批/累计4/跳过4批的例子解释为何要计入 `steps_skipped`。GR恢复测试直接从实际所属 `trainer.py` 导入工厂，不经训练CLI转一层。以上不改变持久化字段或旧检查点键。

仍然较难读的地方：

1. `psi.rhsci.train` 约466行、`lambda0.rhsci.train` 约422行，各自混合参数、配置、预检和执行。可以沿现有职责拆分，但应单独核对原生调用及CLI兼容；本轮没有为缩短文件改动它们。
2. 四个HF `train.py` 保留旧函数重导出以兼容既有导入。查实现应直接进入 `config` / `preflight` / `trainer`，新调用不再借CLI模块访问底层工具；不能直接删除已有导出。
3. `hf_resume` 通过局部函数绑定和一处源码替换适配HF内部循环。比复制整份Trainer更少，但依赖上游内部实现，仍是最需要解释和回归测试的公共代码；没有把它包装成不透明的通用Trainer。
4. Ψ模型/Trainer/数据与GR原生模型内仍有RHSCI接入钩子；仅看包内 `rhsci/model.py` 不足以理解全部前向，参见[架构中的原生接入表](ARCHITECTURE.md#simple-shared-interfaces-20261102)。
5. `args` 与 `paths` 多为Namespace/字典，不同骨干返回项稍异，阅读时需要跳到配置函数。后续新增接口宜明确输入/返回值；不为形式统一把各模型配方塞进一个大型基类。

## 验证与边界

ACT/GR新增真实Trainer恢复测试覆盖19样本、batch2、累计4、checkpoint4以及0/2持久worker，比较样本、最终参数、Adam、调度器和Torch主RNG。π/Ego沿用上轮真实恢复对照。λ真实CPU子进程覆盖正常退出残留rank、SIGTERM/SIGINT、SIGKILL继承锁、写记录失败、Popen未返回/记录回调中的停止，以及退出清理阶段的停止和锁顺序。

本轮CPU回归：π39、GR34、ACT30、Ψ54项通过；λ真实进程6项及既有预算1项、公共CLI11项通过。Ego29项中28项通过，另1项仍因缺少 `flash_attn` 报错；恢复对照通过。GR恢复用例分别在独立HF4.51.3/Accelerate1.10.1与π环境HF4.53.2/Accelerate1.7.0核验，后者已有ceil行为未改变。Ruff按根 `ruff.toml` 范围通过；原生子项目保留各自格式，不混入本轮修改。未正式训练、安装依赖、运行GPU/仿真或读取生产大检查点；PSI0-RAW只同步文档入口，不改生产代码/环境/进程/断点。独立generator不等于保存worker内部随机增强状态，CPU结果不保证全模型多卡逐位等价。仅本地提交，不自动推送。
