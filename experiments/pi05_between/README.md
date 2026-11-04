# π0.5 两桌评测：STATE 与 VGQ-AE

## 新旧包完整首轮与历史首轮逐编号核对（2026-11-04）

旧包复测首轮三级各10次已全部完成，不能再沿用仅共同episode0–7的早期快照。相同权重、当前30/30代码下，旧包成功4/1/1，新包首轮0/1/1；认证历史首轮为4/3/0（其执行30/24、时钟/导航协议不同）。均只比较三级30次，旧包无hard，不是150次最终结果。旧包level0成功episode0/1/3/4，新包均失败；旧level1/2成功均episode1，新包分别episode4/3；历史level0成功4/5/6/7，历史level1成功4/6/9。因此旧包level0在这一轮恢复到历史同样的4/10，但成功身份明显不同；level1未恢复到历史3/10，不能把回归归为一个统一初态平移或据此宣称旧协议已复现。原生噪声未配对，当前旧/新推理服务组织不同，不作显著性或全因果结论。

compare_matched_results.py复用既有官方统计解析器，拒绝任何不足十条/重复/越界的轮次，并核对历史网格。收据matched-first-round-results.json记录六份官方统计及历史审计输入SHA、30条逐编号三方标签和旧/新四格计数；脚本实际运行、Ruff通过，无额外仿真。实际命令：

```bash
cd /root/workspace/RHSCI-Private
python3 experiments/pi05_between/compare_matched_results.py \
  --old-root /opt/rsci-data/evals/pi05-vgq-official-pre1102-singlethread-20261104 \
  --new-root /opt/rsci-data/evals/pi05-between-30x30-5cycles-20261104/VGQ-AE \
  --historical /opt/rsci-data/evals/stutter-root-cause-20261104/historical-pi05-manifests.json \
  --repeat 1 \
  --output /opt/rsci-data/evals/stutter-root-cause-20261104/matched-first-round-results.json
```

## 专家加噪与自主采样路径的条件诊断（2026-11-04）

实际PID2149625已退出0，完成四帧×三seed=12组，每组记录10步；全部加hook与不加hook的原生采样输出逐位相同。flow-conditioning.json及summary/PNG保存在同名运行根，图已人工检查。第30/60/90帧在t=1，当前图的双臂干净动作估计RMSE（三seed均值）0.02679/0.02431/0.02271rad；只替换初始图prefix、保持同一专家加噪输入后为0.06282/0.10030/0.10957rad。模型确实利用视觉条件，不能把问题说成完全忽略图像。第0帧两prefix本来相同，全部teacher指标完全相同。

在t=0.1，四帧专家加噪路径的当前图干净动作估计RMSE依次0.00560/0.00434/0.00414/0.00381rad；自主路径为0.03198/0.01315/0.01972/0.02301rad。最终10步输出对v2训练标签的三seed均值0.03199/0.01315/0.01972/0.02301rad。支持这些样本的自主路径偏离专家路径，后期模型没有将偏差纠回；但teacher误差含t缩放，不能直接将两种路径比较称训练loss异常或某个训练bug已证。全部36维velocity MSE在不同帧/t不相同，不把单个低t误差当原训练loss复现；原日志含Beta抽样/增强/ARDY，当前诊断不含这些步骤。

额外只读导入本机五个patched Transformers模块，Gemma配置/实现、PaliGemma实现、SigLIP check/实现的实际__file__及SHA全部匹配launch_contract，五个声明依赖版本（Torch/Transformers/TorchCodec/Accelerate/SentencePiece）也全部匹配。收据installed-transformers-contract-audit.json包含路径、实际/期望摘要和版本；这缩小了此前“共享当前底层”的未验证范围，不证明所有传递依赖/物理闭环等价。没有生产参数更新、未恢复训练或下载其他权重，GPU1已释放。正式400次和旧150次继续，具体实现根因仍未确证。

此前已证明原训练RGB输入也会产生块边界偏差，但不能据此解释低训练loss的原因。继续核对认证原model.py forward/sample_actions：训练使用x_t=t*noise+(1-t)*actions、target=noise-actions，推理从t=1沿负velocity积分，符号/维度一致。本次只读诊断在现有真实sample_actions投影层加记录hook，逐位检查加记录与不记录的10步输出完全相同；再在同一专家加噪路径各t评估模型，分别用当前图和初始图prefix，观察视觉条件作用。自主路径只报告干净动作估计误差，不错误地给离开专家直线路径的状态套用训练velocity target。

配置来源为本说明已核对论文/README、保存VGQ40k契约及认证历史forward/sample_actions。原训练episode0的0/30/60/90帧，seed17/29/43、输出30、10步eager/BF16、v2原统计不变；无增强、无梯度/训练、无仿真控制/RTC，GPU1额外模型、12GiB空闲预检，不计成功率。此t网格诊断不是训练loss无偏估计（不采Beta分布），四帧不能代表全训练集。实际启动命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
cd /root/workspace/RHSCI-Private
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 python -u \
  experiments/pi05_between/probe_flow_conditioning.py \
  --checkpoint "$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1/checkpoint-40000" \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --recorded-inputs "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/recorded-replay-aligned-inputs-retry/predictions.jsonl" \
  --noise-seeds 17 29 43 \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/flow-conditioning.json" \
  > "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/flow-conditioning.log" 2>&1
```

## 出生点稳定位移与机器人朝向恢复缺陷（2026-11-04）

复用已完成的旧包episode4/execute-30和新版spawn-full/original-episode4轨迹，两者均有160步稳定记录：旧配置x=-0.64261184994m、稳定后-0.70395204182m；新配置-0.70334384994m、稳定后-0.76468403519m。两者自身稳定后退均6.134cm，新旧配置差6.0732cm，稳定后差仍6.0732cm（误差约7nm）。目标相对机器人前向距离由37.515cm增至43.604cm；稳定阶段盒子z均保持约0.76261–0.76311m，没有先掉落。此处证明两段位移叠加，不证明上游未发布生成器重复稳定或其意图有误。官方入口reset后仅一次稳定循环；PR29说明零臂初始化后约6.1cm为正常WBC稳定位移。现有初态视频/parquet的base位姿为占位值，不能由其倒推出正确生成器逻辑。

另外直接调用官方SpatialDR/RobotActor/Pose，复现一个具体读取缺陷：spatial.py机器人分支读取保存的position，却漏读同一state的quaternion；当robot_orientation_region=None时保留Actor默认单位四元数，再覆盖_inner_state中保存的四元数。实际源码SHA51b5f168ff98870d27fad9819b9bcb53cf7fb05c46215d126d72d7a7c3dbcb0e。旧level0十条均单位朝向、误差0；新版四难度40条均保存[0.9999988442451683,0,0,-0.001520364537745943]，却恢复为[1,0,0,0]，全部角误差0.00304073025rad（约0.17422°），位置误差0。真实新旧稳定轨迹的初始四元数也相同到1e-8，支持该路径实际生效。这是已复现的代码缺陷，尚无证据证明如此小的朝向误差解释大幅成功率回归或块边界跳变；只前移6cm的成功反例也未修正这个朝向。未修改官方源码、未执行朝向修复反事实。

只读CPU脚本已实际运行、Ruff通过；报告spawn-settling-audit.json与spatial-robot-restore-audit.json包含输入及实现SHA，输出存在时拒绝覆盖。后者只承载真实uid、不创建机器人物理环境，不是闭环成功率实验。实际命令：

```bash
cd /root/workspace/RHSCI-Private
/opt/rsci-data/.venv-simple/bin/python experiments/pi05_between/audit_spawn_settling.py \
  --old-metadata /opt/rsci-data/eval-official-pre1102/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0/meta/episodes.jsonl \
  --new-metadata /opt/rsci-data/SIMPLE-eval-20261103/snapshot-fb67cd9a/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0/meta/episodes.jsonl \
  --old-trace /opt/rsci-data/evals/stutter-root-cause-20261104/old-episode4-horizon/execute-30/control_trace.jsonl \
  --new-trace /opt/rsci-data/evals/stutter-root-cause-20261104/spawn-full/original-episode4/control_trace.jsonl \
  --episode 4 \
  --output /opt/rsci-data/evals/stutter-root-cause-20261104/spawn-settling-audit.json
/opt/rsci-data/.venv-simple/bin/python experiments/pi05_between/audit_spatial_robot_restore.py \
  --metadata \
  /opt/rsci-data/eval-official-pre1102/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0/meta/episodes.jsonl \
  /opt/rsci-data/SIMPLE-eval-20261103/snapshot-fb67cd9a/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0/meta/episodes.jsonl \
  /opt/rsci-data/SIMPLE-eval-20261103/snapshot-fb67cd9a/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-1/meta/episodes.jsonl \
  /opt/rsci-data/SIMPLE-eval-20261103/snapshot-fb67cd9a/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-2/meta/episodes.jsonl \
  /opt/rsci-data/SIMPLE-eval-20261103/snapshot-fb67cd9a/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-hard/meta/episodes.jsonl \
  --output /opt/rsci-data/evals/stutter-root-cause-20261104/spatial-robot-restore-audit.json
```

## 历史手指兼容的实际组合核对（2026-11-04）

已认证只读取得原bfffc460c0d857ae447f18be2529447ea612a11c的eval_wbc_hand_order.py，SHA f30f8dbe2f9e1aa22e946072865e2520019fb360e991861348e893ee8be5495d。历史协议不仅按名称修正PD，还先对旧模型动作执行legacy_upper_targets；不能遗漏后者、只凭移除具名PD就断言物理目标变了。

复用spawn-full/plus006-episode0的1200个原始策略动作与真实物理位置，加载当前真实MJCF和WBC RobotModel，直接调用原生from_psi0_upper_joints、手部getter及G1Sonic.apply_action方法，与认证历史两层兼容组合比较。给两路同一物理位置、零速度（trace未记录速度，不冒称现场力矩重放），全部43关节PD力矩最大差0Nm。没有推进物理、插值或调用模型，不是闭环反事实。原生手指反馈的确按thumb/middle/index，而WBC契约为thumb/index/middle；本轨迹槽位差最大0.80006rad。但将同一反馈分别经过真实_build_wbc_observation与原生下身compute_observation，1200组86维网络输入最大差0：下身只取body29，插值上身不读反馈。历史旧动作兼容保留原有物理目标，故这个已知手指接口缺陷不能直接当作本次与历史成绩差异的已证根因，也不能因此宣称其他手指控制入口都没有问题。

收据hand-feedback-contract-episode0-verified.json保存实际源路径/摘要、原始trace SHA、名称序列及误差；脚本核验实际导入的下身源路径，Ruff通过。复算命令（输出存在时拒绝覆盖）：

```bash
cd /root/workspace/RHSCI-Private
/root/workspace/Haoxuan/RHSCI-runtime/.venv-simple/bin/python \
  experiments/pi05_between/audit_hand_contract.py \
  --simple-source /opt/rsci-data/external/SIMPLE-official \
  --trace /opt/rsci-data/evals/stutter-root-cause-20261104/spawn-full/plus006-episode0/control_trace.jsonl \
  --output /opt/rsci-data/evals/stutter-root-cause-20261104/hand-feedback-contract-episode0-verified.json
```

## 相机参考帧与策略输入时间诊断（2026-11-04）

实际完成PID2138037，正常退出且原生统计有episode4 False；这是120步截短诊断，不计正式模型失败。trace完整160稳定步＋120策略步、4个30步动作块。左右目各280条参考记录，帧号相邻差均4、Isaac参考时间差均0.0800000038s，callback与直接annotator帧号差均0，world减参考时间均0，图像SHA无相邻重复。模型在0/30/60/90步的输入SHA全部匹配最后稳定观测/策略步29/59/89的左目SHA。排除本样本可见的缓存旧观测或请求错取前一块图像，不把元数据当作像素/骨骼绝对同步证明。MuJoCo每控制步0.02s，Isaac每步多次推进/渲染，两者独立时钟，不将0.08s直接解释成图像延迟。

同一轨迹模型双臂块边界L2为0.11679/0.14184/0.27426rad，块内中位0.010587，边界比13.3971；模型到WBC臂映射最大误差0，观测关节对MJ最大5.55e-7rad。control_analysis.png已人工检查，边界目标回跳及真实肩关节随后反向仍存在。双目MP4各121帧、640×360/50Hz且已闭合。camera_timing_analysis.json保留trace SHA974acc68…及全部请求关联；没有修改生产代码/默认评测参数或宣布顿挫已修复。实际复算命令：

```bash
cd /root/workspace/RHSCI-Private
/root/workspace/Haoxuan/RHSCI-runtime/.venv-simple/bin/python \
  experiments/pi05_between/analyze_camera_timing.py \
  --trace /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/camera-timing-vgq/control_trace.jsonl \
  --output /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/camera-timing-vgq/camera_timing_analysis.json
/root/workspace/Haoxuan/RHSCI-runtime/.venv-simple/bin/python \
  experiments/pi05_between/analyze_control_trace.py \
  --trace /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/camera-timing-vgq/control_trace.jsonl \
  --plot
```

此前关节状态核对不等于证明RGB零延迟。继续沿官方b75源码核对：SonicLocoManipEnv.step先推进MJ、同步Isaac、执行world.step和rep.orchestrator.step，随后_get_obs/_render_frame；IsaacCamera.get_rgba直接读rgb annotator，不走get_current_frame的频率缓存。此源码顺序仍不能代替真实帧时间证据。实验--trace-camera-timing仅配合单episode trace读取本机Isaac4.5已存在的ReferenceTime、对应SWH仿真时间、callback帧、world时间、左右目图像SHA；模型请求另记左目SHA，用来匹配上一观测。不开启额外渲染、不新建材质、不改控制。ReferenceTime与RGB来自同一render product，但元数据不能单独证明每个像素/骨骼绝对同步，需根据实际结果限定结论。

准备并在对话给出：官方旧level0/episode4、VGQ40k/30-30/noRTC/10步BF16/wall、120步，同8147；GPU0、DDS222（未复用220/221/230/231）。论文/README及保存配置同本说明已核对设置，新增项仅测量；不计正式成功率，双目视频保留。实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
probe_root="$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/camera-timing-vgq"
mkdir "$probe_root"
cd "$probe_root"
CUDA_VISIBLE_DEVICES=0 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir "$PSI_RUNTIME_ROOT/eval-official-pre1102/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0" \
  --sim-mode mujoco_isaac \
  --host 127.0.0.1 --port 8147 --headless \
  --num-episodes 1 --episode-start 4 \
  --max-episode-steps 120 --success-criteria 0.4 \
  --eval-dir "$PWD/stats" --save-video \
  --renderer-gpu 0 --dds-domain-id 222 --cpu-threads 8 \
  --trace-control "$PWD/control_trace.jsonl" \
  --trace-camera-timing > "$PWD/console.log" 2>&1
```

## 认证历史训练记录与学习率核对（2026-11-04）

只补读同一已选VGQ-AE权重的小型元数据，共1298264字节，不下载其他权重或optimizer。HF固定ds1x/RHSCI@59a1ea14d4346496d4e55c9fa58ea2f3330fd298，目录checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1：checkpoint-40000/trainer_state.json的SHA69a3230c…、speed_resume_origin.json的SHA477e32b6…均与已验证checkpoint-manifest一致，并核对HF Git blob身份。原文存于运行根evals/stutter-root-cause-20261104/training-history，不写进推理检查点，不把恢复元数据增加为普通推理前提。

训练记录global_step40000、epoch91.3242；action_loss/ardy_loss各4000条（0至39990每10步），总loss/grad_norm/LR各4000条（10至40000每10步），顺序和步号网格完整、数值有限。action_loss在0–999、6000–7999、10000–19999、30000–40000窗口中位数依次0.22844/0.01927/0.01112/0.004833，ARDY同期0.28812/0.17731/0.12575/0.06211。此文件没有eval_指标，不能将低训练loss称为闭环或独立验证通过，也不能单凭日志证明每次优化器更新/样本顺序都正确。

speed_resume_origin声明从原reference运行checkpoint7000迁到execution=both，模型/optimizer/scheduler/trainer_state/八rank RNG通过硬链接保留。进一步核对原train.py SHA83d1c66c…与VGQ契约一致，提取其纯simple_lr_multiplier函数；4000条记录的LR全部精确匹配原公式在log_step-1处的值，最大绝对差0。支持日志中没有调度器重置，不冒称实际读取了optimizer/RNG payload或原训练数据顺序。训练forward输入含专家动作加噪声，低flow-matching误差与自主从噪声生成动作的准确性是不同证据；当前材料不能据此定位具体训练代码bug。

实际只读CPU核对命令（首次不含LR的报告training-history-audit.json保留，完整报告为training-history-lr-audit.json）：

```bash
cd /root/workspace/RHSCI-Private
/root/workspace/Haoxuan/RHSCI-runtime/.venv-simple/bin/python \
  experiments/pi05_between/audit_training_history.py \
  --directory /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/training-history \
  --checkpoint-manifest /root/workspace/Haoxuan/RHSCI-runtime/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1/checkpoint-manifest.json \
  --training-source /root/workspace/Haoxuan/RHSCI-runtime/external/Psi0-pi05-QSAFE-reference/src/rsci_pi05/train.py \
  --output /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/training-history-lr-audit.json
```

## 固定种子预测的共同偏差与波动（2026-11-04）

只分析既有paired-render-training.json（seed17/29/43、同40k/10步BF16/四个时刻），无新增推理/仿真。对三个自然边界的14臂关节接续误差，逐项验证有限样本恒等式：平均平方误差=均值的平方+围绕均值的方差。原训练图与渲染图的合计平方误差中，三样本共同均值偏差占88.40%/95.77%；不是主要由这三个种子彼此的波动贡献。它不是总体条件期望、置信区间或任意噪声都无影响的证明。

三份预测均值的边界L2变化，原训练图为0.15649/0.08488/0.05384rad，渲染图0.22407/0.03981/0.14053rad；专家自然变化为0.00938/0.01278/0.00393rad。因此这批样本不能靠“只是碰巧采到坏噪声”解释。90帧渲染输入的首左肩yaw均值-0.13246、种子间ddof0标准差0.00236rad，原训练输入均值-0.17445、标准差0.00110rad。结合已有最近专家姿态匹配，支持稳定的条件预测偏差，而非仅噪声重新采样；不把最近姿态当成已解码的模型内部阶段。平均动作从未执行，不能当平滑方案或新闭环分数。

完整收据sampling-bias-audit.json包含两份输入SHA和逐边界误差分解。实际CPU命令：

```bash
cd /root/workspace/RHSCI-Private
/root/workspace/Haoxuan/RHSCI-runtime/.venv-simple/bin/python \
  experiments/pi05_between/analyze_sampling_bias.py \
  --predictions /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/paired-render-training.json \
  --recorded-inputs /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/recorded-replay-aligned-inputs-retry/predictions.jsonl \
  --output /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/sampling-bias-audit.json
```

## 完整任务的失败阶段核对（2026-11-04）

只读spawn-full三条完整轨迹，按连续5个策略步满足的几何阈值记录事件：目标原点比首步高0.08m、与篮子XY距离小于0.1m、目标原点高度低于0.1m、机器人XY位移超过0.3m。阈值不是抓取接触分类器或官方成功判定，三路原生噪声未配对。

新版episode4原位失败：1200步最高目标z=0.76832m，初始0.76275m，从未达到抬升阈值；机器人第471步累计移动超过0.3m，盒子最终仍在第一桌（z=0.68872m，不能误写落地），离篮子最近2.10788m。前移6cm的同episode官方True：第105步抬升、第497步进入篮子XY 0.1m范围，749步结束，最近距离0.00750m。前移episode0官方False：第146步抬升、第544步进入篮子附近，第1046步落到z<0.1m，最终z=0.03755m。这说明距离变化可影响抓取，但前移后仍有搬运后掉落的失败，不能合并为一个抓取错误或宣称普遍恢复。

task-stage-audit.json保存完整trace SHA与官方统计原文，PNG三曲线已人工检查。没有启动额外模型或仿真。实际复算命令：

```bash
cd /root/workspace/RHSCI-Private
/root/workspace/Haoxuan/RHSCI-runtime/.venv-simple/bin/python \
  experiments/pi05_between/analyze_task_stages.py \
  /opt/rsci-data/evals/stutter-root-cause-20261104/spawn-full/original-episode4 \
  /opt/rsci-data/evals/stutter-root-cause-20261104/spawn-full/plus006-episode4 \
  /opt/rsci-data/evals/stutter-root-cause-20261104/spawn-full/plus006-episode0 \
  --output /opt/rsci-data/evals/stutter-root-cause-20261104/task-stage-audit.json
```

## 原v2动作标签往返审计（2026-11-04）

只读全91条/58637帧动作及既有24块预测，实际调用当前Processor.normalize，再按正式serve.py同一v2公式反变换。没有启用已撤回的v3，没有改统计、数据或模型。训练统计和输入预测SHA记录在training-label-roundtrip-audit.json，逐维全训练集往返误差与三个诊断episode的自然边界一并保留。

episode0/1/2的双臂标签往返RMSE分别0.00000440/0.00870747/0.00054308rad，最大单值0.000255/0.068875/0.010849rad。不能说v2没有损失；尤其episode1的STATE对原专家RMSE0.01034，改为对实际截断训练标签后为0.00514。VGQ-AE对实际训练标签RMSE仍0.01533/0.03704/0.01039rad；三条轨迹21个自然边界在往返后基本相同或减小，没有产生已观测的预测尖峰。这排除了“这些块边界尖峰由标签反变换直接制造”的解释，不证明截断对学习完全无影响，也不足以证明任意新归一化可修复旧权重。

实际CPU命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
python -u experiments/pi05_between/audit_training_labels.py \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --checkpoint "$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1/checkpoint-40000" \
  --predictions "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/recorded-chunk-inputs/predictions.jsonl" \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/training-label-roundtrip-audit.json"
```

## 三条训练轨迹的离线动作块接续（2026-11-04）

复用已完成的recorded-chunk-inputs/predictions.jsonl，不增加模型/仿真请求。episode0/1/2各8块（0/30/…/210帧），21个自然相邻边界；仅原训练RGB输入。各轨迹VGQ-AE边界变化中位数/块内变化中位数为8.7596/7.6463/5.0911，STATE为2.5987/2.1505/1.6860。VGQ-AE双臂整块专家RMSE为0.01533/0.03861/0.01042rad，STATE为0.00339/0.01034/0.00381rad。每个边界的预测/专家自然变化、上一块终点和下一块起点相对当前真实状态的误差都保留在收据，不仅报告比例。

此结果把“原训练输入也存在接续不连续”的证据从单条扩展到三条轨迹，但不是闭环成功率，也没有固定原生噪声；STATE和VGQ-AE是两份不同训练权重，不能称受控方法消融。部分块的误差随块内时间变化，不能把全部问题当统一常量偏移，再用平移首姿态掩盖。源文件SHA记录在recorded-chunk-boundary-audit.json。现有训练/推理源码中动作分支同用当前帧，历史Reader只用于训练辅助分支，不能把正常的单帧部署直接判成漏传历史的bug。

实际CPU复算命令：

```bash
cd /root/workspace/RHSCI-Private
/root/workspace/Haoxuan/RHSCI-runtime/.venv-simple/bin/python \
  experiments/pi05_between/analyze_recorded_chunks.py \
  --predictions /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/recorded-chunk-inputs/predictions.jsonl \
  --output /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/recorded-chunk-boundary-audit.json
```

## 历史v3导航契约对照（2026-11-04）

已从原归档eval_environment.lock.json确认历史manifest所指协议为legacy-rate-v1-speed-selector-v1，而非纯legacy四维范数。只读原GitHub源码`enterx-x/Psi0@1f96adda:scripts/rsci_between_settings.py:navigation_method`核实：不按vyaw_flag<0.1抑制朝向校正；站走选择用同步前三维速度范数<0.05；返回旧角速度反馈。当前官方入口用flag门控、默认速度+flag范数<0.1，语义确有差异，但是否导致本次动作回跳必须实测。实验legacy_navigation.py只替换原生get_action中这三处，经原生方法SHA保护，并在trace记录实际selector/适配函数SHA；正式入口与控制器磁盘源码不变。

另一并行工作启动的old-episode4-navigation两路180步使用旧包/GPU0/1/DDS220/221。其legacy进程在本轮校正前加载了旧四维范数实现，不代表v3。本轮最初启动的新包navigation-v3碰巧复用了220/221，已只停止自己两路并保留所有目录，current仅稳定阶段、无模型请求；DDS可能交叉干扰，两个目录均不能作为干净对照，不能计入模型失败或原正式分数。实际错误/停止验证见ERROR_LOG。

独立重跑navigation-v3-corrected：新版level0/episode4、VGQ-AE40k服务8147、30/30、无RTC、10步BF16、wall时钟；每组120步只诊断不计成功率，两组均保存双目录像/完整trace。current=GPU2/DDS230，historical-v3=GPU3/DDS231；PID2076437/2076436已正常退出。trace新增真实lower_input/lower_output及原网络选择，便于离线同输入反事实；实验脚本audit_navigation_contract.py仅重用原生数学/分支，对ONNX用占位输出，禁止将其解释为物理回放。实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
probe_root="$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/navigation-v3-corrected"
for spec in 'current 2 230' 'historical-v3 3 231'; do
  (
    read -r variant gpu domain <<< "$spec"
    extra=()
    if [[ "$variant" == historical-v3 ]]; then
      extra+=(--diagnostic-legacy-navigation)
    fi
    mkdir -p "$probe_root"
    mkdir "$probe_root/$variant"
    cd "$probe_root/$variant"
    CUDA_VISIBLE_DEVICES="$gpu" python -u \
      /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
      simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
      psi0_decoupled_wbc train \
      --data-format lerobot \
      --data-dir "$SIMPLE_EVAL_ROOT/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0" \
      --sim-mode mujoco_isaac --host 127.0.0.1 --port 8147 --headless \
      --num-episodes 1 --episode-start 4 \
      --max-episode-steps 120 --success-criteria 0.4 \
      --eval-dir "$PWD/stats" --save-video \
      --renderer-gpu "$gpu" --dds-domain-id "$domain" --cpu-threads 8 \
      --trace-control "$PWD/control_trace.jsonl" "${extra[@]}" \
      > "$PWD/console.log" 2>&1
  ) &
done
wait
```

两路完整120步/4块均已完成，各保留两个已关闭MP4。当前/历史v3的双臂目标边界跳变分别[0.19583,0.14709,0.17575]/[0.09040,0.10676,0.07076]rad，边界/块内比17.9583/9.4920；模型到WBC双臂逐项误差均0，观测对MuJoCo最大误差5.90e-7/9.60e-7rad。恢复历史导航后仍存在模型目标与实际关节回退，图已查看；两组原生随机噪声/图像不成对，不能把比例差当导航收益或成功率。120步的False是截短诊断，不算正式模型失败。

对每一路160次稳定阶段+120次策略阶段的**真实下身插值输入与调用前状态**，分别调用原生get_action和认证历史v3转换；现场实际协议的速度、返回反馈及网络选择逐项完全复现。稳定期两种规则速度/选择全部相同。策略期同输入两种规则的站走选择差0/2次，最大角速度差0.04913/0.06266rad/s。首策略步即有具体差异：当前组输入vyaw_flag=0.00061154、target_yaw=0.01005284时，当前规则转速0，旧规则0.02499918rad/s。当前源码把第三维作为门控flag，历史对照按朝向误差持续校正并返回计算角速度，确有控制语义不兼容；但两种真实闭环均有回跳，不能称它是全部动作问题的充分解释。此CPU对照仅计算命令/分支，ONNX输出被占位且不进入物理环境，不能外推反事实身体轨迹。

复算命令（输出文件存在时拒绝覆盖，重算请选新文件）：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
cd /root/workspace/RHSCI-Private
for variant in current historical-v3; do
  python experiments/pi05_between/audit_navigation_contract.py \
    --trace "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/navigation-v3-corrected/$variant/control_trace.jsonl" \
    --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/navigation-v3-corrected/$variant/navigation-contract.json"
  python experiments/pi05_between/analyze_control_trace.py \
    --trace "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/navigation-v3-corrected/$variant/control_trace.jsonl" \
    --plot
done
```

## 历史录像与导航单因素对照（2026-11-04）

**运行状态更正：** 下方old-episode4-navigation两路与并行navigation-v3短暂复用DDS220/221，存在串扰风险，已TERM本会话PID2056693/2056694、保留产物，整组无效，不报告其跳变比或成功率。并行工作补查到manifest对应历史v3采用前三维速度范数，最初四维范数属于更早导航语义；下方命令仅作这次已作废尝试的溯源，不能称完整历史协议复现。后续导航由已有并行工作负责，本会话不启动重复实例。正式八路及旧包150次通信编号不同，未因此修改或停止。

完整时限出生点对照三路双目视频已人工抽帧核对，full-video-comparison.png保留。成功的plus006-episode4全程边界/块内目标跳变比11.6531，原位失败组11.8459；不得把此前前180步的14.03→6.63外推成全程已消除顿挫。plus006-episode0最终在第二桌前掉落；距离变化影响抓取和任务表现，但不是已经证明的唯一根因或通用修复。

历史原始成功视频自身也有二维手部回撤：已核验SHA的旧level0/episode4录像，第66→72→78帧，扣除选定静态背景点平移后，指尖先移动[1.52,-1.23]px再[-9.28,2.40]px，腕部先[0.49,-1.77]再[-6.85,3.90]px，两段方向余弦-0.911/-0.711。121帧双向光流检查最大误差指尖0.0483px、腕部0.0487px、背景0.1758px，带跟踪点抽帧已人工查看。仅说明选定历史视频窗口也有回撤，不推导三维关节、全部episode或每24帧都反转；单背景平移补偿不消除相机旋转/透视。收据historical-videos/historical-ep4-motion.json，实际复算命令（不启动仿真）：

```bash
cd /root/workspace/RHSCI-Private
/root/workspace/Haoxuan/RHSCI-runtime/.venv-simple/bin/python \
  experiments/pi05_between/track_historical_video.py \
  --video /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/historical-videos/level0-episode4-left.mp4 \
  --output /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/historical-videos/historical-ep4-motion.json \
  --tip 245 188 --wrist 223 252 --reference 95 90
```

继续排除历史导航兼容差异：原enterx-x/Psi0@bfffc460c0d857ae447f18be2529447ea612a11c的scripts/eval_wbc_navigation.py，三处修改恢复旧小角速度/站走选择/返回导航命令。当前原生G1GearWbcPolicy.get_action源码SHA152a1919…与其要求完全相同；实验legacy_navigation.py沿用同一AST转换，要求三处匹配各一次，不复制控制器/ONNX。手指原适配还包含legacy动作兼容，不能仅看具名PD修复就假设旧模型物理手指目标改变；本次导航单因素不动手指。正式入口默认关闭，仅trace诊断允许开启；两路trace均记录真实stand/walk网络调用及命令。

配置先核对本说明的论文/官方README、保存权重与原历史脚本，再明确诊断差异：旧level0/episode4、40k/30-30/无RTC/10步BF16/wall、180步，两路仅导航协议不同，GPU0/1、DDS220/221、同8147。原生随机采样并非固定噪声配对，不计正式分数。对话中已给实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
probe_root="$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/old-episode4-navigation"
for spec in 'official 0 220' 'legacy 1 221'; do
  (
    read -r mode gpu domain <<< "$spec"
    extra=()
    if [[ "$mode" == legacy ]]; then
      extra+=(--diagnostic-legacy-navigation)
    fi
    mkdir -p "$probe_root"
    mkdir "$probe_root/$mode"
    cd "$probe_root/$mode"
    CUDA_VISIBLE_DEVICES="$gpu" python -u \
      /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
      simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
      psi0_decoupled_wbc train \
      --data-format lerobot \
      --data-dir "$PSI_RUNTIME_ROOT/eval-official-pre1102/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0" \
      --sim-mode mujoco_isaac \
      --host 127.0.0.1 --port 8147 --headless \
      --num-episodes 1 --episode-start 4 \
      --max-episode-steps 180 --success-criteria 0.4 \
      --eval-dir "$PWD/stats" --save-video \
      --renderer-gpu "$gpu" --dds-domain-id "$domain" --cpu-threads 8 \
      --trace-control "$PWD/control_trace.jsonl" \
      "${extra[@]}" > "$PWD/console.log" 2>&1
  ) &
done
wait
```

## 训练数据时间索引与预测动作进度审计（2026-11-04）

只读检查实际91条训练轨迹、58637帧：parquet episode/frame identity、有限36维动作/32维状态、视频nb_frames、r_frame_rate/avg_frame_rate与50Hz、逐帧timestamp=frame/50全部通过，最大时间戳误差0。读取裁剪列，不实体化depth、不重建缓存、不改训练数据。它排除了本数据中可观察的容器帧率、帧数、行索引与时间戳错配，不能证明采集源物理RGB/状态绝对同步；腰部状态也不逐项等于上一条动作指令，不能照搬另一个raw转换脚本的字段语义。

同一配对推理结果用14维双臂最近专家姿态作描述（只搜索episode0前240帧，非模型内部状态估计）：seed17四个原训练输入的预测首/末动作最接近专家7→35、24→53、71→101、85→115帧；渲染输入为10→41、21→51、60→86、59→87。90帧渲染输入在seed29/43也分别最接近56→85、56→87帧，而训练输入为85→116、82→113。这个结果更符合“不同块对动作进度定位不一致”，并非所有动作块统一漏/多了固定一帧，不能靠整体平移时间索引解释。最近姿态可能有歧义，不能称唯一时间配准或具体训练代码bug。

完整收据 `evals/stutter-root-cause-20261104/training-timing-phase-audit.json`。实际完成命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
python experiments/pi05_between/audit_training_alignment.py \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --predictions "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/paired-render-training.json" \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/training-timing-phase-audit.json"
```

## 训练原图与专家回放渲染图配对诊断（2026-11-04）

离线对齐episode0第0/30/60/90帧，VGQ-AE忽略state，故成对变化仅图像。同40k权重、30步输出、10步eager/BF16；每对seed17/29/43固定噪声，保存完整双臂动作与专家动作RMSE。复用正式模型及处理器，不改变在线服务；范围只有一个专家轨迹的四个时刻，不能直接外推成功率或训练代码bug。输出 `evals/stutter-root-cause-20261104/paired-render-training.json`，日志同名 `.log`，24次真实推理已正常完成，GPU1释放，脚本Ruff检查通过。

第90帧训练原图首目标左肩yaw为-0.17451/-0.17576/-0.17307rad，回放渲染图为-0.13500/-0.13308/-0.12931rad；该时刻真实回放/专家关节为-0.18602/-0.18968rad，渲染图明显偏向早期。两种输入双臂整块RMSE分别0.01996/0.01832/0.03076与0.05975/0.05859/0.06006rad。但第60帧渲染图反而更准，不能宣布所有回跳由渲染错误引起。所有时刻/seed平均RMSE训练图0.02197、渲染图0.03215rad。

更强的范围判断：完全不接仿真，直接把同seed、0/30/60/90四张原训练RGB分别预测的完整动作块拼接，边界L2变化中位数/块内变化中位数仍为9.3924/7.5048/7.7330；对应渲染图为15.0884/12.7292/14.3769。专家动作三个自然边界仅0.00938/0.01278/0.00393rad，训练图预测边界按seed分别为[0.10229,0.09885,0.03910]/[0.22313,0.07998,0.05303]/[0.15862,0.08144,0.07303]rad。这是离线teacher-forcing输入上的分块不连续，不能当作新的闭环轨迹或全部训练样本；它说明同权重在原训练输入上也存在不连续，不能靠改视频录制/控制器/场景渲染一项解释或消除全部问题。尚未定位具体训练代码bug。

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
cd /root/workspace/RHSCI-Private
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 python -u \
  experiments/pi05_between/compare_inference_modes.py \
  --checkpoint "$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1/checkpoint-40000" \
  --trace "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/expert-replay-vgq/control_trace.jsonl" \
  --recorded-inputs "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/recorded-replay-aligned-inputs-retry/predictions.jsonl" \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --instruction ' pick up the cracker box from table1,locomotion to table2,and place  on table2.' \
  --sampling-steps 10 --noise-seeds 17 29 43 \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/paired-render-training.json"
```

本目录存放本机实验编排，不属于正式训练/推理框架，不作为公开发布入口。数据、权重、环境和输出均在 `/root/workspace/Haoxuan/RHSCI-runtime`。正式推理调用 `rhsci.inference.http_server`，仿真直接调用官方 SIMPLE `run_eval`；没有复制模型或评测循环。

## 论文、README 与本次设置

启动前已阅读 [SIMPLE v2 论文附录 A.3](https://arxiv.org/html/2606.08278v2#A3)、[官方 π0.5 README](https://github.com/physical-superintelligence-lab/Psi0/blob/main/baselines/pi05/README.md)、[SIMPLE decoupled WBC 评测说明](https://github.com/physical-superintelligence-lab/SIMPLE/blob/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11/docs/source/decoupled-wbc/evaluation.md)，再核对保存的权重配置与实际实现。

| 项目 | 本次配置 | 来源与差异 |
| --- | --- | --- |
| 模型 | π0.5 STATE、VGQ-AE，各 checkpoint-40000 | HF RHSCI 已训练权重，不重新训练 |
| 任务 | G1WholebodyLocomotionPickBetweenTablesTeleop-v0 | 官方两桌 Teleop 任务 |
| 预测 / 执行 | 30 / 30，36 维动作，无 RTC | 用户明确指定；权重 horizon=30。论文 π0.5 chunk=16，官方 README 的 SIMPLE 示例 horizon=30，因此不称完全复现论文配方 |
| 采样 | 原生 flow matching，10 步，BF16 autocast | 当前 RHSCI π0.5 推理配方；不将未明确的论文设置推断为相同 |
| 输入 | 当前左目 RGB；STATE 另用 32 维状态；VGQ-AE 推理不读状态/历史 | 保存的模型方法及现有处理器 |
| 归一化 | 恢复训练时配置与匹配的数据统计；不替换为新版统计 | 保存的 contract、rsci_config、训练数据 fingerprint |
| 仿真 | MuJoCo 3.3.6 + Isaac Sim 4.5.0.0，headless | 官方 README 推荐组合；SIMPLE 固定 b75c6786 |
| 控制 | 官方 SONIC decoupled WBC，50 Hz；每块覆盖 0.6 秒仿真时间 | 官方 Teleop 评测说明及控制配置 |
| 初态 | SIMPLE-eval/fb67cd9a，level-0/1/2/hard，各 episode 0–9 | 最新评测资源；论文表格只有 level-0/1/2，hard 为更新后的扩展 |
| 次数 | 每个难度重复 5 轮 × 10 次；每模型 200 次，总计 400 次 | 用户指定；每模型只有 40 个不同初态，重复次数不作为独立初态数 |
| 时限 / 判定 | 1200 控制步；success_criteria=0.4 | 官方任务 metadata，并非从论文推断；保留官方稳定阶段最多 300 步 |
| 随机性 | 原生随机动作采样；状态测量噪声关闭 | 无 RTC；未修改官方客户端注入自定义确定性协议 |
| 输出 | 保存逐 episode 视频、官方统计、异常和每轮命令 | 每轮独立工作目录，避免覆盖 |

本次没有修补官方已知的 WBC/MuJoCo 手指映射问题，沿用固定官方实现，限制见 `docs/SIMPLE_UPDATES_20261103.md`。

## 8 卡分配与实际启动入口

| GPU | 模型 | 难度 | HTTP 端口 | DDS domain |
| --- | --- | --- | --- | --- |
| 0 | STATE | dr-level-0 | 8120 | 100 |
| 1 | STATE | dr-level-1 | 8121 | 101 |
| 2 | STATE | dr-level-2 | 8122 | 102 |
| 3 | STATE | dr-level-hard | 8123 | 103 |
| 4 | VGQ-AE | dr-level-0 | 8124 | 104 |
| 5 | VGQ-AE | dr-level-1 | 8125 | 105 |
| 6 | VGQ-AE | dr-level-2 | 8126 | 106 |
| 7 | VGQ-AE | dr-level-hard | 8127 | 107 |

准备完成后，在仓库根执行：

```bash
cd /root/workspace/RHSCI-Private
bash experiments/pi05_between/launch_8gpu.sh
```

脚本逐项显式指定模型、GPU、难度和 5 轮次数。每个 worker 先启动下面的原生服务并验证一次真实 `/act` 返回有限的 `(30, 36)`，再连续启动 5 个官方评测进程。预热不计入评测次数。

以下是 GPU 0 的实际服务命令；其他卡仅按表替换 GPU、run-dir 和端口：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
export CUDA_VISIBLE_DEVICES=0

python -m rhsci.inference.http_server \
  --backbone pi05 \
  --run-dir "$PSI_RUNTIME_ROOT/checkpoints/pi05-STATE-between-40000" \
  --checkpoint-step 40000 \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8120 \
  --no-deterministic-eval
```

预测/执行 30/30、10 步采样为现有服务的实际固定实现，不虚构不存在的 CLI 参数。VGQ-AE 的 run-dir 为 `$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1`。

GPU 0 第一轮的实际客户端命令如下。`eval_decoupled_wbc.py` 是对官方 `run_eval(EvalConfig(...))` 的薄封装，只额外暴露渲染卡和 DDS domain 参数，使 8 个进程隔离；其余参数与官方命令格式一致：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
export CUDA_VISIBLE_DEVICES=0
export task=G1WholebodyLocomotionPickBetweenTablesTeleop-v0
export result_root="$PSI_RUNTIME_ROOT/evals/pi05-between-30x30-5cycles-20261104"
cd "$result_root/STATE/dr-level-0/repeat-01"

python /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/$task \
  psi0_decoupled_wbc \
  train \
  --data-format lerobot \
  --data-dir "$SIMPLE_EVAL_ROOT/$task/dr-level-0" \
  --sim-mode mujoco_isaac \
  --host 127.0.0.1 \
  --port 8120 \
  --headless \
  --num-episodes 10 \
  --episode-start 0 \
  --max-episode-steps 1200 \
  --success-criteria 0.4 \
  --eval-dir "$PWD/stats" \
  --save-video \
  --renderer-gpu 0 \
  --dds-domain-id 100 \
  --cpu-threads 8
```

每一轮实际执行的绝对路径命令另存于输出目录的 `command.sh`；模型服务命令和 PID 写入 `worker.json`。官方客户端名 `psi0_decoupled_wbc` 表示所用 HTTP 消息协议，并非实际加载 Ψ0 模型。

8 实例并行时显式将每实例 Carbonite/TBB 工作线程限制为 8（Kit `threadCount` / `maxThreadCount`），依据 [NVIDIA 线程配置说明](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html) 处理本机默认每实例数百线程造成的 CPU 争用；这是运行资源配置，不是论文预测/执行设置。此前无上限并行尝试的日志保存在同名 `-before-thread-limit` 输出目录。

## 资源与进度

### 独立加载归档适配层对照（已完成）

在 GPU1 依次独立构造、严格加载归档 STATE 包内适配器和当前适配器，使用同一 VGQ-AE 40k 权重、四块真实 trace、seed17、eager/BF16/10步/30输出。实际进程已正常退出，archived-adapter-parity.json 已生成：四组图像/tokens/mask/actions 处理器张量逐项相同，完整30×36归一化及物理动作最大差均0、输出逐位相同。两次模型分别构造和加载，仍共享当前安装的原生骨干代码；归档源码来自STATE包，认证VGQ源码范围以稍后单列对照为准。该实验不改变在线服务或正式评测。实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python -u \
  experiments/pi05_between/compare_archived_inference.py \
  --checkpoint "$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1/checkpoint-40000" \
  --legacy-source "$PSI_RUNTIME_ROOT/checkpoints/pi05-STATE-between-40000/source-code/src/rsci_pi05" \
  --trace "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/vgq-observation/control_trace.jsonl" \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/archived-adapter-parity.json"
```

### 根因排查继续：旧评测初态对照（用户更正）

另按专家回放相同时刻0/30/60/90读取原训练RGB，调用当前STATE/VGQ服务作训练图与渲染图对照。每图一次原生随机采样，不能当固定噪声配对或唯一根因证据。原训练图上STATE双臂整块RMSE依次0.00450/0.00413/0.00307/0.00350rad，VGQ为0.02705/0.03864/0.02677/0.02144rad；相应专家回放渲染图上VGQ为0.01865/0.01355/0.00921/0.05985rad，方向并不一致，不能从最后一帧误差较大就认定渲染根因。首次沙箱HTTP被拒，没有发出请求，保留空产物；提权后另目录完成。实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
python -u experiments/pi05_between/probe_recorded_inputs.py \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/recorded-replay-aligned-inputs-retry" \
  --episodes 1 --frames 0 30 60 90 --state-port 8120 --vgq-port 8126
```

出生距离单因素补充：旧level0 episode4机器人x=-0.64261184994，新=-0.70334384994，后移0.060732m；目标位置两版均[-0.32878145576,-0.01041561365,0.76342171431]。先前训练初态+0.06及新版+0.12将盒子碰落，不是“新版只恢复这6厘米”的反例。已用prepare_layout_probe.py从新版level0建立 `eval-probes-20261104/new-level0-x-plus006`（--robot-x-offset 0.06），只改10条dr_state_dict.spatial.g1_sonic.position[0]，其余数据软链接/metadata保持；probe_source.json保留来源与变化。以下两组同一新版episode4、当前30/30/无RTC/10步BF16/wall时钟、各180步、原生随机采样，诊断不计成功率、不换官方数据，除x之外全部相同。来源为官方新旧metadata、保存的权重设置及上方README。实际命令：

180步两组已完成：原位/前移的边界跳变比14.0284/6.6259，仍非完全消除；模型到WBC双臂误差均0。两组启动时目标z均约0.76275m，没有稳定阶段碰落目标；前移组180步目标z=0.90724m，视频120步已提起，原位组180步z=0.76281m仍在桌面。逐帧对照图 `new-episode4-spawn/spawn-video-comparison.png` 已人工查看。训练91初态目标0相对机器人出生x距离min/median/max=0.30450/0.31899/0.33537m，旧level0十初态0.30783/0.31870/0.33182m，新level0为0.36856/0.37944/0.39256m。这是初始配置距离而非稳定后实测距离；单条随机采样对照只能支持距离变化为影响因素，不证明唯一根因或任务成功。

源头补查：新版自带meta/episodes.jsonl.pre-fk中robot配置在level0/1共20条与旧包全部相同，level2有9条相同；新包30条中29条机器人统一偏移[-0.060732,-0.002489333333,0]，另一条x额外后移。上游SIMPLE-org/data的00bab43e（2026-10-30）提交标题为eval-fixed: shifted-spawn eval levels ... for 12 teleop tasks，11月2日38e43f22再同步到psi-data；不是材质恢复操作改了出生点。运行根spawn-metadata-audit.json记录metadata摘要、所有距离/偏移及上游API证据。自带视频first frame与当前稳定后图像姿态也有区别，但它本身不证明重复稳定或某行导出代码错误；stored-frame-comparison.png保留作后续证据。

据此继续完整1200步诊断：新版episode4原位/前移、episode0前移，GPU0/2/3、DDS217/218/219，同8147，其他配方不变。保持官方成功判定，不以提起盒子代替两桌完整成功；不混入正式400次。来源同本节，实际命令（在对话中也已给出）：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
probe_root="$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/spawn-full"
for spec in 'original 4 0 217' 'plus006 4 2 218' 'plus006 0 3 219'; do
  (
    read -r variant episode gpu domain <<< "$spec"
    data_dir="$SIMPLE_EVAL_ROOT/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0"
    if [[ "$variant" == plus006 ]]; then
      data_dir="$PSI_RUNTIME_ROOT/eval-probes-20261104/new-level0-x-plus006"
    fi
    mkdir -p "$probe_root"
    mkdir "$probe_root/$variant-episode$episode"
    cd "$probe_root/$variant-episode$episode"
    CUDA_VISIBLE_DEVICES="$gpu" python -u \
      /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
      simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
      psi0_decoupled_wbc train \
      --data-format lerobot --data-dir "$data_dir" \
      --sim-mode mujoco_isaac \
      --host 127.0.0.1 --port 8147 --headless \
      --num-episodes 1 --episode-start "$episode" \
      --max-episode-steps 1200 --success-criteria 0.4 \
      --eval-dir "$PWD/stats" --save-video \
      --renderer-gpu "$gpu" --dds-domain-id "$domain" --cpu-threads 8 \
      --trace-control "$PWD/control_trace.jsonl" \
      > "$PWD/console.log" 2>&1
  ) &
done
wait
```

下方是此前180步诊断实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
probe_root="$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/new-episode4-spawn"
for spec in 'original 2 215' 'plus006 3 216'; do
  (
    read -r variant gpu domain <<< "$spec"
    data_dir="$SIMPLE_EVAL_ROOT/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0"
    if [[ "$variant" == plus006 ]]; then
      data_dir="$PSI_RUNTIME_ROOT/eval-probes-20261104/new-level0-x-plus006"
    fi
    mkdir -p "$probe_root"
    mkdir "$probe_root/$variant"
    cd "$probe_root/$variant"
    CUDA_VISIBLE_DEVICES="$gpu" python -u \
      /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
      simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
      psi0_decoupled_wbc train \
      --data-format lerobot --data-dir "$data_dir" \
      --sim-mode mujoco_isaac \
      --host 127.0.0.1 --port 8147 --headless \
      --num-episodes 1 --episode-start 4 \
      --max-episode-steps 180 --success-criteria 0.4 \
      --eval-dir "$PWD/stats" --save-video \
      --renderer-gpu "$gpu" --dds-domain-id "$domain" --cpu-threads 8 \
      --trace-control "$PWD/control_trace.jsonl" \
      > "$PWD/console.log" 2>&1
  ) &
done
wait
```

已找到原GitHub真实历史证据，不再仅依赖STATE快照：`enterx-x/Psi0` 的 `feture/ARDY_PI05_QSAFE@1369f3e9a94ba3112935db3047715e74713d6c13` 只读稀疏检出在运行根external/Psi0-pi05-QSAFE-reference，model.py=5a6cc3…、processing.py=93150b…与本VGQ权重launch_contract完全匹配，原gemma_pytorch也匹配4f75d3…。按下方适配层完整命令，将archived-source改为 `$PSI_RUNTIME_ROOT/external/Psi0-pi05-QSAFE-reference/src/rsci_pi05`、output改为 `evals/stutter-root-cause-20261104/authenticated-adapter-comparison.json`，实际已运行完成；四张真实图像的处理器全部输出逐项相同，30×36归一化/物理动作最大绝对差均0。两版仍共用当前原生骨干，结论限定认证适配层数值对照，不证明所有原生环境历史行为一致。

历史视频manifest树SHA `e95f4b0a3681b74c81dc817b5ac5641b3672ee9f`，全部60条Git blob身份验证、权重SHA51f037…均与当前同一模型。前两轮三级成功7/7/0、完成20/20/20，即14/60，不外推全五轮。manifest的stats_sha256是官方 **results/eval_stats.txt成绩文件**，不是归一化统计，上传器rsci_now_videos.py第54行可证；不能因它与训练stats.json不同就宣称归一化错配。原历史服务 `scripts/rsci_pi05_server.py` 明确返回actions[:24]，评测receipt记录action_chunk30/action_execute24，无RTC；此外原协议启用sim-step时钟、手指具名旧动作兼容和legacy导航。故30/30的新包/旧包对照不能称原历史全栈复现。只读审计命令（不下载额外权重）：

```bash
python experiments/pi05_between/audit_historical_results.py \
  --tree e95f4b0a3681b74c81dc817b5ac5641b3672ee9f \
  --checkpoint-sha256 51f0379bdbf0f072ef067387b14e16b9b97b4b90848989d0266d5e7e4ecc5947 \
  --output /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-root-cause-20261104/historical-pi05-manifests.json
```

已仅下载历史level0第一轮episode0失败/episode4成功的双目4视频，共约8.6MB，逐文件SHA匹配manifest。first120抽帧人工查看，两者初始双手接近桌沿，episode4持续向盒子靠近；该有限抽帧不证明全片没有顿挫，也不单凭画面推导关节角。文件与拼图在运行根 `evals/stutter-root-cause-20261104/historical-videos`。

据真实旧设置增加独立单因素执行长度诊断：旧level0/episode4（历史成功），同一40k VGQ服务、10步BF16、无RTC、当前官方wall控制，每组180步，分别预测30执行30/24，仅trace入口允许截取前缀，正式入口默认不变。原生随机采样、两次渲染不是固定噪声配对，不能把单次改善当严格因果成功率。来源为上述历史实际服务/receipt，诊断不冒称论文设置、不计正式分数。实际启动命令：

两路实际完成180步。30/30与30/24的双臂目标边界/块内跳变比为12.3423/12.2444，边界跳变中位数为0.13375/0.12744rad（14维联合L2），没有消除回跳。逐块验证预测均30步、执行分别精确取前30/24步，实际块起点30/60/90/120/150与24/48/72/96/120/144/168；模型到WBC双臂最大误差均0，观测对MuJoCo最大误差5.90e-7/5.75e-7rad。完整统计及轨迹图在上述两路control_analysis.{json,png}，双目视频保留。用同目录analyze_control_trace.py --trace <各路control_trace.jsonl> --plot可重算。正式评测仍30/30，本次短探针的failed标签不纳入成功率。

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
probe_root="$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/old-episode4-horizon"
for spec in '30 0 213' '24 1 214'; do
  (
    read -r horizon gpu domain <<< "$spec"
    mkdir -p "$probe_root"
    mkdir "$probe_root/execute-$horizon"
    cd "$probe_root/execute-$horizon"
    CUDA_VISIBLE_DEVICES="$gpu" python -u \
      /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
      simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
      psi0_decoupled_wbc train \
      --data-format lerobot \
      --data-dir "$PSI_RUNTIME_ROOT/eval-official-pre1102/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0" \
      --sim-mode mujoco_isaac \
      --host 127.0.0.1 --port 8147 --headless \
      --num-episodes 1 --episode-start 4 \
      --max-episode-steps 180 --success-criteria 0.4 \
      --eval-dir "$PWD/stats" --save-video \
      --renderer-gpu "$gpu" --dds-domain-id "$domain" --cpu-threads 8 \
      --trace-control "$PWD/control_trace.jsonl" \
      --diagnostic-execute-horizon "$horizon" \
      > "$PWD/console.log" 2>&1
  ) &
done
wait
```

适配层生产尺寸对照：同一已下载VGQ-AE 40k真实权重、四张实际闭环输入、seed17噪声，归档与当前Processor/模型采样各运行10步BF16输出30×36，均eager，不改在线服务。归档Processor SHA93150b…与VGQ训练contract完全匹配；模型源码SHA7946e1…来自STATE快照，与VGQ contract的5a6cc3…不匹配，必须保留这个范围限制。两版共享当前原生骨干及同一权重张量，故覆盖的是适配层方法及处理器，不是两套独立原生依赖或认证历史VGQ源码。来源设置同本说明已核对的论文/README与保存权重；固定噪声/双eager仅用于数值诊断。实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
CUDA_VISIBLE_DEVICES=0 python -u experiments/pi05_between/compare_inference_modes.py \
  --checkpoint "$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1/checkpoint-40000" \
  --trace "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/vgq-observation/control_trace.jsonl" \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --archived-source "$PSI_RUNTIME_ROOT/checkpoints/pi05-STATE-between-40000/source-code/src/rsci_pi05" \
  --instruction ' pick up the cracker box from table1,locomotion to table2,and place  on table2.' \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/archived-adapter-comparison.json"
```

最新启动更正：原独立服务三客户端在第一条200后发生cuDNN `CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED`，随后CUDA Graph捕获错误/500，三路技术中止，不计模型False。初始失败发生于AnyIO不同请求线程的图捕获；线程局部CUDA上下文/内存池是待核解释，不能仅凭此认定原周期顿挫根因。已停止该服务与残留的一路仿真，保留全部失败日志及目录。实验入口 `serve_single_thread.py` 仅将既有server.predict_action通过单工作线程调用，其他CLI/模型/HTTP协议复用正式实现，生产框架及原八路未改。新服务PID1801072预热1次＋三客户端并发6次全200且有限(30,36)，证据 `evals/stutter-root-cause-20261104/shared-server-singlethread-check.json`；不能将短请求验证当作完整150次已通过。

实际修复服务命令（已执行）：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 \
TORCHINDUCTOR_CACHE_DIR="$PSI_RUNTIME_ROOT/cache/torchinductor-pi05-gpu7" \
nohup python -u experiments/pi05_between/serve_single_thread.py \
  --backbone pi05 \
  --run-dir "$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1" \
  --checkpoint-step 40000 \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --device cuda:0 --host 127.0.0.1 --port 8147 \
  --no-deterministic-eval \
  > "$PSI_RUNTIME_ROOT/evals/old-vgq-singlethread-server.log" 2>&1 &
nohup bash experiments/pi05_between/run_old_eval.sh 0 4 8147 210 \
  > "$PSI_RUNTIME_ROOT/evals/old-singlethread-level0-launch.log" 2>&1 &
nohup bash experiments/pi05_between/run_old_eval.sh 1 5 8147 211 \
  > "$PSI_RUNTIME_ROOT/evals/old-singlethread-level1-launch.log" 2>&1 &
nohup bash experiments/pi05_between/run_old_eval.sh 2 6 8147 212 \
  > "$PSI_RUNTIME_ROOT/evals/old-singlethread-level2-launch.log" 2>&1 &
```

其余配置来源同下方完整旧包方案：40k、30/30、无RTC、10步BF16、三等级各50次、前两轮双目录像。最终输出根 `evals/pi05-vgq-official-pre1102-singlethread-20261104`；下方dedicated输出保留为技术失败尝试，不能当成功率结果。历史元数据归档 `eval-metadata-and-provenance-20261027.tar.zst/historical-eval/RSCI_EVALUATION.md` 明确一组45/150属于OFFICIAL-Pi05-RGB，且后来标记状态协议不匹配；不能冒充用户所述VGQ-AE约30%的对应记录。

用户进一步明确：对照官方仅level-0/1/2、11月2日同步新版之前的完整评测包；VGQ-AE历史成功率约30%为用户提供，历史权重/命令尚未核实。已下载官方 `USC-PSI-Lab/psi-data/e78fb93cc28912a3031a10b8656d32d7f0a2b867/simple-eval` 的两桌ZIP，480532 bytes、SHA256 `8a4b0ff7d068967dac8baec2c4a4deca4f644215a1ee15e921739f3dfe90e7e0`。官方 `38e43f22` 在2026-11-02 07:50:12 UTC同步eval-fixed。旧官方包与此前RHSCI镜像包ZIP容器摘要不同，但解包后75个文件逐一SHA256相同、全部30条environment_config相同。之前120步探针用了同一旧内容，但不足以判断完整成功率。

完整旧包复测沿用上方论文/官方README核对：VGQ-AE 40k、用户指定30/30/无RTC、原生10步采样/BF16、保存的模型统计、官方任务1200步/0.4判定，三个等级各10初态×5轮=150次，前两轮全部双目录像。当前SIMPLE代码b75c6786不回退，这是评测包单因素对照，尚非历史全栈复现。独立模型服务GPU7/8147，GPU4/5/6额外仿真、独立DDS210/211/212。首次复用8124–8126的准备进程已终止并保留日志，原因是其服务受原worker生命周期控制，可能早于旧包5轮结束而退出；技术中止不计模型失败。最终实际启动命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
CUDA_VISIBLE_DEVICES=7 \
TORCHINDUCTOR_CACHE_DIR="$PSI_RUNTIME_ROOT/cache/torchinductor-pi05-gpu7" \
nohup python -u -m rhsci.inference.http_server \
  --backbone pi05 \
  --run-dir "$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1" \
  --checkpoint-step 40000 \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --device cuda:0 --host 127.0.0.1 --port 8147 \
  --no-deterministic-eval \
  > "$PSI_RUNTIME_ROOT/evals/old-vgq-dedicated-server.log" 2>&1 &
nohup bash experiments/pi05_between/run_old_eval.sh 0 4 8147 210 \
  > "$PSI_RUNTIME_ROOT/evals/old-dedicated-level0-launch.log" 2>&1 &
nohup bash experiments/pi05_between/run_old_eval.sh 1 5 8147 211 \
  > "$PSI_RUNTIME_ROOT/evals/old-dedicated-level1-launch.log" 2>&1 &
nohup bash experiments/pi05_between/run_old_eval.sh 2 6 8147 212 \
  > "$PSI_RUNTIME_ROOT/evals/old-dedicated-level2-launch.log" 2>&1 &
```

`run_old_eval.sh` 内逐轮完整原生入口即实际执行配置；每轮另存effective_eval_config.json、eval.log和官方stats。官方Kit可能提前结束Python，result.json有则保留，不能仅凭退出码判断完成。运行根产物为 `evals/pi05-vgq-official-pre1102-dedicated-20261104`，不混入新包统计；旧包无hard，汇报level0/1/2，hard标不适用。尚未完成，不将120步截短诊断计为模型失败。

固定输入seed17的10/20/50步积分对照已完成：step30左肩yaw首目标-0.03072/-0.03261/-0.03268 rad，实际-0.11909；双臂首目标误差L2为0.22313/0.22245/0.22126 rad。增加积分步数没有消除回退，不改正式10步，也不据此排除所有训练/兼容问题。

独立 eager solver 诊断：固定同一输入与 seed17 噪声，比较10/20/50步积分，30步输出/BF16不变。10步是正式设置，其余只是数值诊断，不改在线服务，不将其称论文要求；GPU1额外加载已下载同一权重，12GiB可用显存预检同原后端对照。实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
CUDA_VISIBLE_DEVICES=1 python -u \
  experiments/pi05_between/compare_inference_modes.py \
  --checkpoint "$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1/checkpoint-40000" \
  --trace "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/vgq-observation/control_trace.jsonl" \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/sampling-step-comparison.json" \
  --instruction ' pick up the cracker box from table1,locomotion to table2,and place  on table2.' \
  --sampling-steps 10 20 50
```

固定同一批真实评测图像，交换当前/初始状态及当前/初始图像，每模型每条件原生随机重复3次，检验反馈信息，不能把不同权重间比较当同权重消融。使用同一 STATE 权重的状态交换是单变量输入对照。指令取匹配训练 meta/tasks.jsonl 的 task0 文本（旧 trace 尚未记录 instruction，故显式传入）；模型 40k/30输出/10步/BF16 同本节来源。只推理，不控制仿真：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
python -u experiments/pi05_between/probe_feedback.py \
  --trace "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/vgq-observation/control_trace.jsonl" \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/feedback-comparison.json" \
  --state-port 8120 --vgq-port 8126 --repeats 3 \
  --instruction ' pick up the cracker box from table1,locomotion to table2,and place  on table2.'
```

追加专家动作回放与仿真观测旁路推理：数据 50Hz，每次连续 30 步专家动作控制当前仿真，VGQ-AE 40k/10步/BF16 仅旁路预测并记录，与原训练状态/专家动作比较。当前官方 wall 控制、原训练完整初态不变；轻量 fixture 与训练配置逐项相同。120 步诊断，不属于策略成功率。命令分别在两个进程执行：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
python experiments/pi05_between/prepare_layout_probe.py \
  --source "$SIMPLE_EVAL_ROOT/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0" \
  --environment-from "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --output "$PSI_RUNTIME_ROOT/eval-probes-20261104/training-x-zero" \
  --robot-x-offset 0
python -u experiments/pi05_between/replay_expert_policy.py \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/expert-policy" \
  --port 8140 --compare-port 8126
```

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
mkdir -p "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/expert-replay-vgq"
cd "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/expert-replay-vgq"
CUDA_VISIBLE_DEVICES=6 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir "$PSI_RUNTIME_ROOT/eval-probes-20261104/training-x-zero" \
  --sim-mode mujoco_isaac \
  --host 127.0.0.1 --port 8140 --headless \
  --num-episodes 1 --episode-start 0 \
  --max-episode-steps 120 --success-criteria 0.4 \
  --eval-dir "$PWD/stats" --save-video \
  --renderer-gpu 6 --dds-domain-id 206 --cpu-threads 8 \
  --trace-control "$PWD/control_trace.jsonl"
```

WBC 仿真时钟完整闭环诊断：保留新版原始 level2/episode3，只显式把两个原生 agent 模块和复合 WBC 的时间引用设为当前 MuJoCo 时间加同一 epoch，覆盖稳定与策略两阶段；不改标准库全局时间、HTTP、模型或资产。复合策略 last_goal_time 必须同域，避免伪造 timeout。默认仍为官方 wall；simulation 仅允许单 episode 且必须 trace，日志 changes_control=true。本组用于检验原先只 CPU 重放策略动作未覆盖的初始化影响，未宣称修复。实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
mkdir -p "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/sim-clock-vgq"
cd "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/sim-clock-vgq"
CUDA_VISIBLE_DEVICES=6 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir "$SIMPLE_EVAL_ROOT/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-2" \
  --sim-mode mujoco_isaac \
  --host 127.0.0.1 --port 8126 --headless \
  --num-episodes 1 --episode-start 3 \
  --max-episode-steps 120 --success-criteria 0.4 \
  --eval-dir "$PWD/stats" --save-video \
  --renderer-gpu 6 --dds-domain-id 205 --cpu-threads 8 \
  --control-clock simulation \
  --trace-control "$PWD/control_trace.jsonl"
```

新版 level2/episode3 只前移出生 x 0.12m 的独立对照，测试新版出生点后移与当前稳定阶段后退叠加的假设。光照、材质、目标和干扰物均用原新版配置；其余配置同本节其他 120 步诊断，不是官方推荐初态，不混入正式结果。实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
python experiments/pi05_between/prepare_layout_probe.py \
  --source "$SIMPLE_EVAL_ROOT/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-2" \
  --output "$PSI_RUNTIME_ROOT/eval-probes-20261104/new-x-plus012" \
  --robot-x-offset 0.12

source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
mkdir -p "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/new-x-plus012-vgq"
cd "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/new-x-plus012-vgq"
CUDA_VISIBLE_DEVICES=7 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir "$PSI_RUNTIME_ROOT/eval-probes-20261104/new-x-plus012" \
  --sim-mode mujoco_isaac \
  --host 127.0.0.1 --port 8126 --headless \
  --num-episodes 1 --episode-start 3 \
  --max-episode-steps 120 --success-criteria 0.4 \
  --eval-dir "$PWD/stats" --save-video \
  --renderer-gpu 7 --dds-domain-id 204 --cpu-threads 8 \
  --trace-control "$PWD/control_trace.jsonl"
```

本轮结果（均为 120 步截短诊断，不计正式成功率）：

专家动作回放已完成：同一执行链在30/60/90步的双臂目标跳变仅0.00938/0.01278/0.00393rad，边界/内部比0.97，没有额外换块尖峰；真实关节相对原训练状态的RMSE依次为0.000827/0.002758/0.006237rad。原始RGB与当前同初态回放图像已并排人工查看，场景/机器人基本重现，未发现可确认的左右目或相机挂载错误；不能把前面对不同DR初态的肉眼差异当相机不一致。专家回放临时服务8140已正常退出。

固定同一真实评测图像（第30步）做输入交换，以下为3次原生采样均值：真实左肩偏航为-0.11909rad；STATE当前状态预测-0.12127±0.00102，只替换为初始状态则为-0.00777±0.00082；STATE保留当前状态、图像换初始图仍为-0.12239±0.00096。VGQ-AE当前图像预测-0.03031±0.01178，初始图像为-0.00602±0.00105。第60/90步重复该模式。误差不是一次随机采样可以解释的量级，但三次样本不构成正式显著性检验。

这组证据将本样本的故障收窄到视觉条件策略对当前动作进度的反馈不足：已执行的关节状态是正确的、原始图像在更新，VGQ-AE新块却仍退向较早姿态；STATE同权重状态交换可复现或解除这种回退。STATE和VGQ-AE是不同权重，不能把两者比较当作同权重的训练方法消融，也没有读取到模型内部“估计的关节角”。VGQ-AE按现有训练设计本就不在推理时使用状态/历史，不能擅自补状态token冒充修复。具体训练层成因仍未确认，不把以上结果写成某行训练代码bug或“模型已修复”。原始证据分别为 `expert-replay-comparison.json`、`feedback-comparison.json` 与对应完整trace。

| 输入初态/配置 | 三次双臂块边界 L2 跳变 / rad | 边界中位 / 块内中位 |
| --- | --- | --- |
| 新版 level2/episode3，VGQ-AE，前轮记录 | 0.236 / 0.288 / 0.209 | 23.04 |
| 实际旧版 level2/episode3，VGQ-AE | 0.073 / 0.119 / 0.187 | 13.08 |
| 新版同初态，仅灯光设训练值 50000 | 0.213 / 0.191 / 0.238 | 20.74 |
| 原训练 episode0 的完整初态 | 0.086 / 0.016 / 0.150 | 8.79 |
| 原训练初态出生 x+0.06（盒子在稳定阶段落桌，非有效恢复） | 0.158 / 0.195 / 0.164 | 17.07 |
| 新版初态出生 x+0.12（盒子在稳定阶段落桌，非有效恢复） | 0.185 / 0.252 / 0.148 | 18.34 |
| 新版原始初态，WBC 全程仿真时钟 | 0.257 / 0.250 / 0.305 | 25.12 |

旧数据已真实运行，但未消除回跳；灯光单因素被反例排除为充分解释。训练初态显著减弱但未消除回跳，不能把它写成已修复。旧/新 episode3 完整配置递归比较有 272 个字段差异，收据 `evals/stutter-root-cause-20261104/layout-comparison.json`。灯光副本 10 个 episode 共 120 个字段只改了同一灯光的 layout/dr_state_dict 强度，没有增减材质或物体。

必须保留的反例：两组前移初态都在策略接管前将盒子碰落桌，不能当作良好输入恢复。官方 [PR29 原文](https://github.com/physical-superintelligence-lab/SIMPLE/pull/29) 明确零臂初态后的约 -0.061m 是 WBC 自身 settle（当前 b75c678 已含该修复），不能把本轮观测到的 6.1cm 直接当新 bug。仿真时钟组 WBC 在恰好 30/60/90 步的跳变虽仅约 0.01rad，但下一步恢复完整跳变，全 120 步最大 WBC 双臂跳变 0.3045，最大实际关节单步变化 0.0277rad，未修复周期反向。不能只挑边界单帧报成功。

另核对原始 HF 资产清单的全部 g1_sonic 及 graspnet/objects：7008 文件逐项 SHA256 与当前相同（机器人68文件）；未替换、增减或修改资产。结合前轮 scene102344022/vMaterials 的6014文件相同，已覆盖本样本机器人、场景、目标物资源身份，但不代表任务配置或渲染版本必然相同。收据 `robot-object-comparison.json`。

两组离线训练图像诊断已完成：15 个分散帧，STATE 双臂专家目标 RMSE 中位 0.00391，VGQ-AE 0.00983 rad；另 24 个与块时刻对齐帧，中位分别 0.00391/0.01129 rad。VGQ-AE 对 episode0 的训练 30/60/90 帧首动作左臂偏航 -0.0948/-0.1698/-0.1655，接近专家 -0.0885/-0.1407/-0.1700；证明不是无论图像如何都机械重放初态，不代表所有离线样本/闭环均正确。完整专家/预测动作与 RGB 在 `recorded-inputs`、`recorded-chunk-inputs`。采样未固定噪声，不能将两组差异当严格配对结论。

后端源码进一步核对：归档 STATE 训练快照的 gemma_pytorch、Gemma 模型/配置、PaliGemma、SigLIP 五文件与当前逐字节相同；当前环境实际导入的三份 Transformers replacement 与仓库文件相同，收据 `backend-source-comparison.json`。该证据不冒充取得 VGQ-AE 训练源码完整快照或全模型原/现数值配对。

稳定阶段位移的单变量诊断：训练初态原 x=-0.644018，当前原生稳定后约 -0.70539，后退约 6.1 cm。新评测出生点本身又比旧/训练初态后移约 6 cm。不能未经验证就改正式评测；独立把训练 episode0 出生 x 加 0.06，其他完整 environment_config 与训练数据逐项相同。官方评测单帧 parquet 仅作 loader 的轻量承载，动作仍来自现有模型，不播放该 parquet 动作。此前直接读取训练数据时，原生 get_episode_lerobot 逐帧实体化深度图导致准备耗时数分钟；py-spy 确认 hf_transform_to_torch，非模型或 DDS 卡死。原训练初态对照最终已正常完成。

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
python experiments/pi05_between/prepare_layout_probe.py \
  --source "$SIMPLE_EVAL_ROOT/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-0" \
  --environment-from "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --output "$PSI_RUNTIME_ROOT/eval-probes-20261104/training-x-plus006" \
  --robot-x-offset 0.06

source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
mkdir -p "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/training-x-plus006-vgq"
cd "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/training-x-plus006-vgq"
CUDA_VISIBLE_DEVICES=6 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir "$PSI_RUNTIME_ROOT/eval-probes-20261104/training-x-plus006" \
  --sim-mode mujoco_isaac \
  --host 127.0.0.1 --port 8126 --headless \
  --num-episodes 1 --episode-start 0 \
  --max-episode-steps 120 --success-criteria 0.4 \
  --eval-dir "$PWD/stats" --save-video \
  --renderer-gpu 6 --dds-domain-id 203 --cpu-threads 8 \
  --trace-control "$PWD/control_trace.jsonl"
```

离线输入追加与换块时刻对齐的真实训练帧，其他配置同下方第一组，仍不属于闭环成功率：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
python -u experiments/pi05_between/probe_recorded_inputs.py \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/recorded-chunk-inputs" \
  --episodes 3 --frames 0 30 60 90 120 150 180 210 \
  --state-port 8120 --vgq-port 8126
```

实际检查：91 条训练轨迹 lighting intensity 均为 50000；新版 level-0 同值，level-1/2/hard 和旧版 level-1/2 约 8000–12000。此时尚不能据此判为根因。单变量反事实只在独立 layout 副本把 layout/dr_state_dict 的同一灯光强度改为训练值，不改变材质、初态或物体；其他数据软链接到官方源。该配置是诊断偏离，绝不标作官方配置或正式成功率。复用 GPU6 模型服务，GPU7 渲染、DDS202；其余来源/配置同上。

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
python experiments/pi05_between/prepare_layout_probe.py \
  --source "$SIMPLE_EVAL_ROOT/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-2" \
  --output "$PSI_RUNTIME_ROOT/eval-probes-20261104/light-50000" \
  --light-intensity 50000

source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
mkdir -p "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/light-50000-vgq"
cd "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/light-50000-vgq"
CUDA_VISIBLE_DEVICES=7 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir "$PSI_RUNTIME_ROOT/eval-probes-20261104/light-50000" \
  --sim-mode mujoco_isaac \
  --host 127.0.0.1 --port 8126 --headless \
  --num-episodes 1 --episode-start 3 \
  --max-episode-steps 120 --success-criteria 0.4 \
  --eval-dir "$PWD/stats" --save-video \
  --renderer-gpu 7 --dds-domain-id 202 --cpu-threads 8 \
  --trace-control "$PWD/control_trace.jsonl"
```

再用训练 episode 0 的完整 environment_config 作当前渲染/原始训练视频对照，排除仅难度不同导致的外观差异。配置来源、30/30 无 RTC、BF16/10 步不变；120 步仅诊断、不计正式评测。实际命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
mkdir -p "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/training-layout-vgq"
cd "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/training-layout-vgq"
CUDA_VISIBLE_DEVICES=6 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --sim-mode mujoco_isaac \
  --host 127.0.0.1 --port 8126 --headless \
  --num-episodes 1 --episode-start 0 \
  --max-episode-steps 120 --success-criteria 0.4 \
  --eval-dir "$PWD/stats" --save-video \
  --renderer-gpu 6 --dds-domain-id 201 --cpu-threads 8 \
  --trace-control "$PWD/control_trace.jsonl"
```

另做真实训练记录的离线输入诊断：复用 `SimpleRHSCIDataset.raw_sample` 解码前 3 条训练轨迹各 5 个时刻，当前左目 RGB 和实际状态分别送现有 STATE/VGQ-AE 服务，保存完整专家/预测动作。沿用上文官方 README 与权重配置的输入处理、30 步输出、10 步采样和 BF16；VGQ-AE 仍不使用状态。每个输入每模型一次原生随机采样，不声称固定噪声配对；这是训练样本诊断，不是留出成功率。启动命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
cd /root/workspace/RHSCI-Private
python -u experiments/pi05_between/probe_recorded_inputs.py \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/recorded-inputs" \
  --episodes 3 \
  --state-port 8120 \
  --vgq-port 8126
```

上轮仅定位到动作块回跳，没有查清其更上游根因，不能据此结束 root-cause goal。用户所说的旧评测数据包括 task-layouts 的初态、布局、相机与资源引用，不是只比较 USD/材质。已从本机 HF `dataset-ea9d480f/eval-data/SIMPLE/task-layouts/G1WholebodyLocomotionPickBetweenTablesTeleop-v0.zip` 独立解包到运行根 `eval-old-layouts-20261104`；本机该归档 SHA256 为 `ed40763fd5d2d23e121c2a2900c31c30ed70823b8ee7310bb66907adb92f2527`，与旧更新审查记录的另一旧 ZIP 身份不同，不能混称同一份旧包。

level-2/episode-3 旧机器人 x=-0.630689，新 x=-0.703344；目标位置、干扰物和部分 metadata 也不同。沿用本说明已核对的论文/README 来源、同一 VGQ-AE 40k、30/30、无 RTC、10 步 BF16，独立 DDS 200，只截取 120 控制步作诊断，视频和原始输入/动作完整保存，不混入正式成功率。实际启动命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
mkdir -p "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/old-layout-vgq"
cd "$PSI_RUNTIME_ROOT/evals/stutter-root-cause-20261104/old-layout-vgq"
CUDA_VISIBLE_DEVICES=6 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir "$PSI_RUNTIME_ROOT/eval-old-layouts-20261104/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-2" \
  --sim-mode mujoco_isaac \
  --host 127.0.0.1 --port 8126 --headless \
  --num-episodes 1 --episode-start 3 \
  --max-episode-steps 120 --success-criteria 0.4 \
  --eval-dir "$PWD/stats" --save-video \
  --renderer-gpu 6 --dds-domain-id 200 --cpu-threads 8 \
  --trace-control "$PWD/control_trace.jsonl"
```

用户反馈动作卡顿后追加诊断：复用 GPU 6 现有 VGQ-AE 模型服务（端口 8126），另启独立仿真和 DDS 196，不加载第二个大模型；level-2 episode 3，预测/执行 30/30、无 RTC、10 步采样均不变。仅截取 120 控制步以记录前 4 块，保留视频，属于诊断而非正式成功率；`trace_control.py` 只包装原生方法记录每块动作、WBC 目标/输出、仿真时间和真实时间，不更改控制时钟。实际命令：

后续两个输入对照沿用同一完整命令和 120 步，只替换输出目录、GPU/服务/DDS：`state-observation` = GPU 0、port 8120、DDS 197；`vgq-observation` = GPU 6、port 8126、DDS 198。新增记录每块实际输入状态/图像，以及 MuJoCo 与观测关节角；不更换场景资源。原诊断已完成 120 步，4 块推理约 0.068–0.082 秒，每步仿真 0.02 秒；双臂块间目标跳变 0.212/0.203/0.284 rad 的联合 L2，块内中位 0.010 rad，约 21 倍。CPU 原生插值器重放证明：换仿真时钟只会将边界尖峰后移一帧，不能解决模型目标回跳，不能据单个边界帧变小就声称时钟修复有效。

三个诊断均已完成。输入对照中 STATE 的三次双臂边界跳变为 0.0224/0.0726/0.0404 rad，块内中位 0.00982 rad，边界中位约 4.12 倍；VGQ-AE 为 0.2364/0.2880/0.2086 rad，块内中位 0.01026 rad，约 23.04 倍。三份 trace 每个已执行动作的原始模型双臂值与 WBC 输入逐项完全相同，最大误差为 0，排除本样本的双臂队列错位或转换引入回跳。观测与 MuJoCo 双臂最大角度误差分别为 1.17e-6 / 5.66e-7 rad；实际输入图像已保存并人工查看，图像会随姿态更新。VGQ-AE 左肩 yaw 在第 29 步目标约 -0.116 rad，下一块起点约 -0.022 rad，真实关节随后也反向。这定位了已复现卡顿的直接机制：30 步块接续时模型目标不连续；尚不能据此断言权重、训练或所有输入兼容细节均无问题，也没有将其归因为已证实的场景资源错误。

两个输入对照的初始状态逐项相同，但独立仿真渲染的初始图像并非逐像素相同，不能称严格同输入模型配对。VGQ-AE 相邻请求图像像素平均绝对差为 7.55/2.45/1.40（0–255），STATE 为 7.74/11.32/10.03；这是图像有更新的证据，不替代相机全链路时序验收，收据 `input-check.json`。归档 STATE 源码的 `encode/expert/sample_actions` AST 与当前一致，收据 `inference-source-comparison.json`；该归档不是经过身份核验的 VGQ 源码，也不替代全模型数值对照。

未改变模型、30/30、RTC、正式评测、控制时钟或场景材质；没有加平滑掩盖回跳。根据用户最新顺序，原版场景资源替换仍作为最后手段，当前未执行。截短诊断的视频 failed 标签不计正式失败或成功率。仅解码/帧数正确不足以验证动作连续性，此前视频格式复查不能替代本次控制轨迹检查。

数值与图在运行根 `evals/stutter-diagnostic-20261104/{original,state-observation,vgq-observation}/control_analysis.{json,png}`，输入拼图为同级 `input-comparison.png`。重算命令（simple 环境提供 NumPy/Matplotlib，无需安装新依赖）：

```bash
cd /root/workspace/RHSCI-Private
/root/workspace/Haoxuan/RHSCI-runtime/.venv-simple/bin/python \
  experiments/pi05_between/analyze_control_trace.py \
  --trace /root/workspace/Haoxuan/RHSCI-runtime/evals/stutter-diagnostic-20261104/vgq-observation/control_trace.jsonl \
  --plot
```

原始 120 步诊断的完整启动命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
mkdir -p "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/original"
cd "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/original"
CUDA_VISIBLE_DEVICES=6 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir "$SIMPLE_EVAL_ROOT/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/dr-level-2" \
  --sim-mode mujoco_isaac --host 127.0.0.1 --port 8126 --headless \
  --num-episodes 1 --episode-start 3 \
  --max-episode-steps 120 --success-criteria 0.4 \
  --eval-dir "$PWD/stats" --save-video \
  --renderer-gpu 6 --dds-domain-id 196 --cpu-threads 8 \
  --trace-control "$PWD/control_trace.jsonl"
```

视频保留规则：每个模型、每个难度保留 repeat-01 / repeat-02 的全部 10 个 episode，左右目均保存（每模型两轮共 160 个 MP4）。repeat-03–05 不录像，统计/日志/命令照常保留，不删除已有视频。`run_evaluation.py --video-rounds 2` 默认落实该规则，客户端显式接收 `--evaluation-round N --video-rounds 2`；为兼容仍在运行的旧 worker，客户端也可从 `repeat-NN` 工作目录推断轮次，无需中断当前评测。客户端在 Kit 导入前保存 `effective_eval_config.json` 和 `VIDEO_POLICY`，实际 `save_video` 以此为准。当前运行中的第一轮已按旧入口开启录像，之后新启动的每轮客户端读取新规则。

追加独立编译数值对照：GPU 1，复用 VGQ-AE 权重和四张真实输入，固定噪声 seed=17，BF16、10 步采样、30 步输出，对比 eager 与 max-autotune；不接控制器，不计 episode，不更改在线服务。来源仍为本节已核对的保存配置与原 README；指令显式使用匹配训练数据 `meta/tasks.jsonl` 的 task-0 文本，不声称该独立对照重放了在线服务的原随机噪声。12 GiB 可用显存预检不通过则拒绝启动。实际命令：

对照已正常结束。四张输入上双臂最大单关节误差 0.004512 rad、最大联合 L2 误差 0.009599 rad；有 BF16 执行差异，不是逐位相同。eager 第 30 步左肩下一块起点为 -0.03072 rad，此时实际观测为 -0.11909 rad，关闭编译后仍有明显反向目标。结果 `inference-mode-comparison.json`；不能把候选内核 cc=120 错误当成这次推理失败，完整输出已返回。额外进程已退出，显存已释放。

同时只读核对 RHSCI 原资源 manifest：`hf/RHSCI/dataset-ea9d480f/eval-data/SIMPLE/assets-20261016/assets-manifest.json` 中当前 episode 场景 `102344022` 与全部 `vMaterials_2` 的 6014 个文件，与现用资源文件名集合及 SHA256 完全一致，缺少/多出/内容不同均为 0；收据 `original-scene-comparison.json`。这限定于该场景和材质，不替代原物体/机器人资产的比较。本轮没有更换资源或声称完成资源替换 A/B，也没有修改模型/RTC/控制时钟/30 步执行配置；诊断定位直接机制，并非已经修复某个已确认的代码错误。

独立编译数值对照的完整启动命令：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
TORCHINDUCTOR_COMPILE_THREADS=4 \
TORCHINDUCTOR_CACHE_DIR="$PSI_RUNTIME_ROOT/cache/stutter-compile-check" \
python -u experiments/pi05_between/compare_inference_modes.py \
  --checkpoint "$PSI_RUNTIME_ROOT/hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1/checkpoint-40000" \
  --trace "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/vgq-observation/control_trace.jsonl" \
  --dataset "$PSI_RUNTIME_ROOT/data/processed/G1WholebodyLocomotionPickBetweenTablesTeleop-v0" \
  --tokenizer "$PSI_RUNTIME_ROOT/models/paligemma_tokenizer.model" \
  --instruction " pick up the cracker box from table1,locomotion to table2,and place  on table2." \
  --output "$PSI_RUNTIME_ROOT/evals/stutter-diagnostic-20261104/inference-mode-comparison.json"
```

实际录像路径为 `模型/难度/repeat-NN/data/evals/psi0_decoupled_wbc/任务/难度/episode_N/head_stereo_{left,right}_{success,failed}.mp4`。没有 success/failed 后缀的文件可能仍在写入，不能把未关闭的视频算作完整产物。

结果始终按 level-0 / level-1 / level-2 / hard 排序，分别报告成功数和已完成数向量；例如成功 `0/0/1/0`、已完成 `4/4/4/4`，不能将未运行样本算失败。使用只读命令：

```bash
python3 experiments/pi05_between/summarize_results.py \
  --output-root /root/workspace/Haoxuan/RHSCI-runtime/evals/pi05-between-30x30-5cycles-20261104
```

本次复查已核对官方动作队列为空才请求下一块、upsample_factor=1、每 episode reset 清队列与历史标志，当前日志持续收到 30 步动作；统计与已关闭视频 success/failed 后缀一致。成功/失败样本各一份完整 FFmpeg 解码通过，成功视频抽帧已人工检查。未发现新致命异常或材质引用错误，不据此宣称不存在任何 bug；官方篮子空碰撞网格和手指映射限制仍保留。

固定模型 revision `59a1ea14d4346496d4e55c9fa58ea2f3330fd298`，训练数据来源为用户指定 dataset revision `ea9d480ff1d95f34b63d7ea368b1bc72e353fd57`。下载完成核对 HF SHA256，再恢复模型配置、训练统计和 tokenizer；模型不需要 optimizer/RNG 即可推理。

运行根为 `/root/workspace/Haoxuan/RHSCI-runtime/evals/pi05-between-30x30-5cycles-20261104`。`worker-gpu*.log` 为编排日志，每卡目录有 `server.log`，每轮有 `console.log`、`stats/` 和视频。Kit 关闭可能提前退出 Python，因此每轮客户端退出后额外核验官方 `stats/eval_stats.txt` 中 episode 0–9 恰好各一条，再生成 `verified_results.json`；不以关闭之后可能未写出的 `result.json` 或退出码 0 代替。仅有 `warmup.json` 不能视为物理评测完成；每卡五轮均验证共 50 次后才写 `COMPLETE.json`。

## 材质与 USD 引用检查

2026-11-04 追加按官方 ZIP 原始清单复核：49 个场景包与 1 个材质包，共 34068 个文件，文件名集合缺少 0、多出 0、大小/ZIP CRC32 不符 0；88 处 props/textures 别名与其对应官方目录文件集合和内容一致。归档 SHA256 在恢复时已验证，本次文件内容比较为大小与 CRC32，不将其称为再次全量 SHA256。官方材质包本身含有嵌套目录，按原样保留。命令为 `python experiments/pi05_between/audit_material_inventory.py`，完整结果在运行根 `logs/material-inventory-audit.json`。

全 40 初态的实际 reset/render 与桌面 MDL 绑定检查现使用下列命令，结果逐初态写入 `logs/pi05-between-material-rendering-all40/progress.json`，全部通过后才生成 `result.json`；这是资源诊断，不计入正式评测。检查有效绑定的 MDL 路径和导出名称，不能把“MDL 文件存在”当作绑定正确。官方保存的 `ground_material` 字段在当前 DR 实现中没有应用，保持官方行为，没有自行额外绑定地面材质。

完整结果已通过：四难度各 10 个初态共 40 个真实渲染、80 处桌面/第二桌面绑定核对，composition errors 为 0，材质 MDL 路径/导出名称与官方 layout 一致，未绑定的第二桌面未多绑定；日志无缺失 USD/纹理/UsdToMdl/traceback。收据为上述目录的 `result.json`，诊断进程已正常退出。篮子碰撞 mesh 错误仍按官方已知限制保留，不将其误报为材质错误。

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
CUDA_VISIBLE_DEVICES=7 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/check_material_rendering.py \
  --all-episodes
```

```bash
python3 experiments/pi05_between/check_materials.py
```

已静态检查 40 个初态中的 80 处材质使用：63 个不同 MDL 文件及其导出材质名称、175 个不同纹理路径全部存在。收据在运行目录 `logs/pi05-between-material-paths.json`。此检查不替代实际 USD/MDL 加载；客户端开启 `SIMPLE_ISAAC_LOG_LEVEL=warning` 保留引用警告，检查各轮 `console.log`。

实际渲染进一步复现了 `/root/tmp/<编号>/props/*.usd` 缺失引用：场景含向上多层的相对路径，解析后命中指向旧目录的失效链接。`strings` 无法可靠读取 USD crate 的压缩引用；修复脚本使用 USD API 读取引用、解析路径，并检查/恢复 88 处 props 与 textures 兼容别名。两者必须同时恢复，否则 USD 组合可正常但纹理仍缺失。旧链接改名保留，官方 USD 不修改。命令如下，收据为 `logs/pi05-between-usd-aliases.json`：

```bash
uv run --python /root/miniforge/bin/python3 --with usd-core --no-project \
  python experiments/pi05_between/repair_usd_paths.py

source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
CUDA_VISIBLE_DEVICES=7 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/check_material_rendering.py
```

四难度各 episode 0 的真实 reset/render 诊断结果写入 `logs/pi05-between-material-rendering/result.json`，不计入评测次数；需同时检查 warning 日志与图像。恢复环境时也已修正 vMaterials_2 额外嵌套目录。篮子空碰撞 mesh 报错不属于材质缺失，保留该已知问题记录；具体错误证据与当前验证状态维护在 `docs/ERROR_LOG.md`。

## 单卡双路容量试跑

结果：第二路模型真实预热通过；Isaac 场景加载阶段 GPU 2 峰值 30336 MiB，余量降到 1774 MiB，触发既定 2 GiB 保护，第二路停止并清理，原来的 8 路继续。尚未完成这一个诊断 episode，因此不能宣称双路完整评测已通过；没有自动扩大到每卡两路。此结果表明当前两套独立服务/仿真组合的余量很紧，不是一次已经发生的 OOM。原生 Kit 明确记录约 300 秒的 RTX RtPso 着色器编译等待；未通过降画质、删材质来回避峰值。

用户要求检查单卡能否同时跑两套完整评测。原 8 路继续运行；在 GPU 2 额外启动独立 STATE 服务、Isaac、MuJoCo、WBC，端口 8130、DDS 110，输出 `evals/dual-gpu2-capacity-probe-20261104`。配置仍为 checkpoint-40000、30/30、无 RTC、1200 步、阈值 0.4、保存视频；只跑 level-0 的 1 个初态作为容量诊断，不计入正式每模型 200 次。原生流程与论文/README 对照不变。

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh pi05
cd /root/workspace/RHSCI-Private
python -u experiments/pi05_between/probe_dual_gpu.py
```

探针实际启动如下完整 worker，每两秒记录 `memory.jsonl`；余量低于 2048 MiB 时只停止第二路，并保留 `result.json`。不因 `/health` 或预热成功就扩展为全卡双路。

```bash
python -u experiments/pi05_between/run_evaluation.py \
  --gpu 2 --slot 1 \
  --variant STATE --level dr-level-0 \
  --num-repeats 1 --num-episodes 1 \
  --output-root /root/workspace/Haoxuan/RHSCI-runtime/evals/dual-gpu2-capacity-probe-20261104
```
