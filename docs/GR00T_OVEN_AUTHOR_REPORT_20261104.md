# GR00T 烤箱评测复现信息与作者咨询稿

快照时间：2026-11-04 09:48:49 UTC。以下为本机真实配置和结果；STATE尚未完成，数值不是最终结果。本稿供用户向SIMPLE作者咨询，未代发任何消息。

## 可直接发给作者的问题

我们在 SIMPLE 的 `G1WholebodyOpenOvenTeleop-v0` 上，以 GR00T N1.6 做单任务微调和评测。RGB＋本体状态模型在当前快照仅成功1/124次，去掉真实状态、加入我们自己的VGQ-AE训练方法后成功80/150次。两者共享初始化、同一训练数据指纹、训练步数和评测接口，但并非只有是否输入state这一项不同，因此不能把差异直接归因于state。

使用11月2日合并前的三级旧评测包，SIMPLE源码为当前固定b75c6786，而不是整个历史软件栈；没有使用四级新版包。想请教以下设置是否与贵组对应实验一致，以及是否有已知兼容问题：

1. 论文附录A.3的GR00T “released checkpoint”具体是哪一份、哪个revision？公开训练预设的 `pretrain_he_g1_h1_mixed_scratch_gr00t/checkpoint-50000` 与 NVIDIA `GR00T-N1.6-3B` 是什么关系？不能仅凭目录名确认论文初始化。
2. 原始RGB baseline是否使用下述32D状态、下述左手指序、腰部r/p/y，以及“上一次身高命令”而非实测高度？是否有额外状态/时间对齐要求？
3. 训练有效16步、模型内部50步、部署截取前16步并执行完再取新图，4次去噪、无RTC，是否正确？
4. 旧初态包搭配b75c6786及decoupled_wbc/gear_sonic当前固定版本，是否会产生已知动作、腰部、导航、初始化姿态或成功判定差异？建议使用哪些源码/控制器/场景版本组合？
5. 烤箱对应训练集、成功判定和门铰链阈值是否有过修正？我们使用repaired-training-20261015单任务数据，贵组表8烤箱则是16任务联合训练，是否有可比较的单任务结果？
6. 附带level0/repeat1/episode0双目视频中，机器人正前下方有边缘整齐的黑色区域，STATE/VGQ-AE同初态都有。该区域是否为预期渲染？尚未定位到具体prim/材质/阴影，不能认定正常或已修复。

## 当前成绩

向量顺序固定为level0 / level1 / level2 / hard；NA表示旧包没有该等级，不是0分。

| 模型 | 成功数 | 已完成数 | 已完成部分的成功率 |
| --- | --- | --- | --- |
| GR00T STATE | 1/0/0/NA | 40/41/43/NA | 2.50% / 0% / 0% / NA |
| GR00T VGQ-AE | 34/23/23/NA | 50/50/50/NA | 68% / 46% / 46% / NA |

STATE总计1/124，尚在运行；VGQ-AE总计80/150。每级10个固定初态重复5轮，每级最终50次，不能称50个独立初态。初次准备中止、启动故障不计为模型失败。此表从当前成功运行输出中的官方 `eval_stats.txt` 逐episode读取，进行中轮只计已写入True/False的episode，不把剩余episode算失败。

| 模型/等级 | 第1轮 | 第2轮 | 第3轮 | 第4轮 | 第5轮 |
| --- | --- | --- | --- | --- | --- |
| STATE L0 | 0/10 | 0/10 | 1/10 | 0/10 | 未有完成记录 |
| STATE L1 | 0/10 | 0/10 | 0/10 | 0/10 | 0/1，进行中 |
| STATE L2 | 0/10 | 0/10 | 0/10 | 0/10 | 0/3，进行中 |
| VGQ-AE L0 | 6/10 | 5/10 | 7/10 | 7/10 | 9/10 |
| VGQ-AE L1 | 4/10 | 4/10 | 3/10 | 6/10 | 6/10 |
| VGQ-AE L2 | 2/10 | 5/10 | 5/10 | 6/10 | 5/10 |

论文表8的OpenOven GR00T数值为85%/60%/90%，但来自16任务联合微调、每级20次，不能与我们本次单任务结果当作同协议复现对照。论文表1没有烤箱单任务条目。

## 版本与资源身份

| 项目 | 实际版本 |
| --- | --- |
| SIMPLE源码 | `b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11`，跟踪文件git status干净 |
| decoupled_wbc子模块 | `5b304e4d643fe89cd000068dcbba7e7519539caa` |
| gear_sonic子模块 | `e61eeb1dec9678dcc02f97c5ac27ca6a4fe82128` |
| GR00T-WholeBodyControl子模块 | `948022a6fd19bd4ad10186d587b632482453b9a2` |
| 旧评测包 | `USC-PSI-Lab/psi-data@e78fb93cc28912a3031a10b8656d32d7f0a2b867` |
| 旧包文件 | `simple-eval/G1WholebodyOpenOvenTeleop-v0.zip`，348202字节 |
| 旧包SHA256 | `7be16a067d796a5bf5b6a7eed2350e43085c0f72a12dd302cba6875c4d99063e` |
| 独立场景/机器人/材质资产库 | `USC-PSI-Lab/SIMPLE@0e8f23cce8f2e96717aa60a0e26bed9d56223c2d` |
| 烤箱场景资源归档 | `scenes_hssd_102344280.zip`，SHA256 `88b9852978687e6c67865a601ec4b4292fd45964b69b8b6476730b15afa91181` |
| RHSCI模型仓库 | `ds1x/RHSCI@59a1ea14d4346496d4e55c9fa58ea2f3330fd298` |
| 模型checkpoint | 两模型均 `checkpoint-20000`，本地完整归档SHA256已核对HF |
| 本地适配仓库HEAD | `a5c464a750b3d7151fc5740d4d33b7406afcaa85`；另有未提交实验/文档修改，不能称工作树逐字等于该HEAD |

权重归档路径：

```text
STATE:
checkpoint/Gr00t/STATE/oven/priority-rgb-reader-20261028-v1/checkpoint-20000.tar.zst
SHA256: 5099de589cd7067e2d428183fb8ffbf1d60f48356dd2b8d8bf04eda79d49eaf8

VGQ-AE:
checkpoint/Gr00t/VGQ-AE/oven/priority-rgb-reader-20261028-v1/checkpoint-20000.tar.zst
SHA256: 7ef0b8a6ef9e2e109bc1d748e15e87d1c7054d2a81b76c15837c7a275b0ae6f2
```

这些是我们训练的RHSCI模型，不是下载SIMPLE作者训练好的烤箱baseline直接评测；STATE采用GR00T原生状态条件路径，VGQ-AE包含我们自己的方法。

## 训练记录：来自保存契约与源码快照

| 项目 | STATE | VGQ-AE |
| --- | --- | --- |
| 初始化 | NVIDIA GR00T-N1.6-3B | 同左，配置/权重索引SHA相同 |
| 任务 | 单任务oven | 单任务oven |
| 数据配方名 | `simple-oven-repaired-training-h264-20261015` | 同一RGB/动作数据，另有ARDY sidecar |
| 数据规模 | 保存源码配方为99条、37903帧、50Hz | 同左 |
| 数据指纹 | `0cb3eb2c41a1038a497643dc751cef0ef5b69364f81c5fb3549c44317b0233fd` | 相同 |
| 更新次数 | 20000 | 20000 |
| 训练并行/批次 | world_size4 × batch6，累计1，全局24 | 相同 |
| 种子 | 292285 | 292285 |
| 训练视觉帧偏移 | [0] | [-6,-4,-2,0]，50Hz |
| 真实本体状态 | 有 | 无 |
| VLM | top4，视觉编码器冻结，调LLM顶层4层 | 相同 |
| Reader | 无，single-frame-native | 每帧8 image＋8 text query |
| 辅助损失权重 | 0 | ARDY 0.1 |
| 有效动作长度 | 16 | 16 |

两者保存的优化配置：AdamW fused，lr1e-4、betas(0.9,0.999)、eps1e-8、weight_decay1e-5、max_grad_norm1.0、cosine、warmup_ratio0.05。训练worker4。颜色增强brightness0.3、contrast0.4、saturation0.5、hue0.08；按seed/epoch/sample确定增强随机性。

数据规模/来源这里是保存配方证据：本机尚未下载该烤箱训练数据逐帧复核，不将此描述为新完成的数据内容验收。模型初始化契约路径及初始化config/index SHA可在原始launch_contract.json查阅。公开SIMPLE训练预设的另一处本地初始化路径与论文实际权重的关系未确证。

## 推理输入与预处理

视觉输入是头部左目RGB，原始640×360 uint8；右目仅录像，不作为第二个模型视角。语言使用episode中的任务指令，例如level0 episode0为 `move forward to the oven and open it`，processor会小写并删除标点。

STATE和VGQ-AE部署均使用当前帧。VGQ-AE的训练历史帧与Reader/辅助目标属于其训练方法；部署没有把仿真真实本体状态输入VGQ-AE。客户端仍可能在HTTP包中携带states，但空state modality分支不读取它。不能把两个变体描述为仅切换一个state开关。

原生eval processor：Albumentations SmallestMaxSize短边256（INTER_AREA）→固定宽高各95%的中心裁剪→再次短边256→Eagle原生视觉处理。95%指线性宽高比例，不是保留95%面积；评测没有随机裁剪/颜色抖动。训练对应随机位置的95%裁剪和上述颜色增强。没有额外叠加π0.5诊断中的裁剪补丁。

STATE原始32维布局如下；HTTP形状(1,32)，分组送给GR00T原生processor：

| 0基索引 | 内容 | 单位/来源 |
| --- | --- | --- |
| 0:3 | 左拇指 | joint_qpos[29:32]，rad |
| 3:5 | 左中指 | joint_qpos[34:36]，rad |
| 5:7 | 左食指 | joint_qpos[32:34]，rad |
| 7:14 | 右手7关节 | joint_qpos[36:43]，rad |
| 14:21 | 左臂7关节 | joint_qpos[15:22]，rad |
| 21:28 | 右臂7关节 | joint_qpos[22:29]，rad |
| 28:31 | 腰roll/pitch/yaw关节角 | joint_qpos[[13,14,12]]，rad；不是浮动基座世界姿态 |
| 31:32 | 上一次执行的base_height_command | m；每episode初始0.74，不是实测基座高度 |

不包含腿部关节、关节速度、全局位置、物体位姿、接触力或阶段真值。WBC另有完整机器人反馈，不等于VLA只输入这些就让控制器也丢弃下身状态。

归一化沿用各权重statistics.json，use_percentiles=false，使用min/max映射，state超范围裁到[-1,1]；未重算/替换统计。apply_sincos_state_encoding虽为true，但sin_cos_embedding_keys=null，因此本配置没有指定组进行sin/cos扩维；按32个归一化状态值再补零到模型max_state_dim128。VGQ-AE不输入真实状态，模型固定形状填充不代表真实状态反馈。

## 动作表示、采样与控制

| 项目 | 实际值 |
| --- | --- |
| 两份权重动作表示 | 10组全部ABSOLUTE |
| 模型内部action_horizon | 50，max_action_dim128 |
| 训练有效动作位置 | 0–15 |
| HTTP返回 | 前16步×36维，float动作 |
| 客户端执行 | 16步全部入队，执行完再请求下一块；upsample_factor默认1 |
| 去噪次数 | 4，来自保存模型config |
| 推理精度/模式 | BF16，model.eval()/processor.eval() |
| RTC | 不使用；同步请求，没有推理与动作执行重叠 |
| 随机性 | `--no-deterministic-eval`，未启用配对固定噪声；训练seed不等于逐episode评测seed |
| WBC时间 | 官方time.monotonic墙钟；未开启sim-clock诊断 |

36维动作前32维与上述state组序相同，但表示目标命令；后4维为torso_vx、torso_vy、torso_vyaw、target_yaw。命名沿用数据/模型，不自行推定第3个导航量在各历史控制器中的语义等价；客户端将32:36原样交给当前官方navigate_cmd处理。上身经官方from_psi0_upper_joints排列变换，腰部命令映射yaw=30/roll=28/pitch=29，身高=31。没有给输出再次加当前关节状态。

STATE保存use_relative_action=true，VGQ-AE=false，但原生相对变换同时要求action_config.rep=RELATIVE；本次10组均ABSOLUTE，所以两者均未减/加state。CPU使用真实STATE配置/统计，仅切换开关比较apply_action/unapply_action，10组最大输出差均0；此检查不证明全部训练链路正确。

下身沿用官方psi0_decoupled_wbc→SonicDecoupledWbcAgent→decoupled_wbc控制链；物理MuJoCo、视觉Isaac Sim。任务metadata physics_dt0.005s，control_hz200，render_hz50；高层动作以50Hz仿真步执行，16步名义覆盖0.32秒仿真时间。同步推理等待与渲染开销使墙钟时间更长，不声称实时50Hz。未启用legacy-navigation、固定噪声、控制trace或相机时间诊断。

## 评测初态、终止与视频

三级各episode0–9、每级五轮；使用旧ZIP完整environment_config，包括场景、物体、灯光、材质、机器人初态，不只是替换场景USD。这里的场景/材质指配置与资源引用，不能误读为旧ZIP内含全部USD/纹理二进制。该ZIP实际75个文件，类型为json/jsonl/parquet/mp4，没有usd/usda/usdc/png/jpg/mdl/obj/stl资产；实际场景资产从上表独立资源库解析。已核对本机场景336个非目录文件与对应原ZIP清单/大小/CRC一致，无缺失、额外或不匹配；ZIP总341项包含目录。尚未证明这份独立资产库与作者“旧版评测”当时实际资产逐文件相同，不能称整套场景资源已回滚至历史版本，需向作者确认推荐版本/是否曾改动。CLI split=train是官方LeRobot初态加载接口名称，不代表将训练轨迹作为评测视频直接回放。

每episode最多1000步，约20秒仿真动作时间。success_criteria=0.5来自当前官方任务metadata，不是50%成功率/50%开门角度。当前任务代码：任一articulate_joint_1或articulate_joint_2的qpos>1.4rad视为开门；满足时reward每次compute_reward累加0.03，不满足则清零；check_success在reward>=0.5时返回True。不额外假设每个仿真步只调用一次compute_reward，因此不把累计次数擅自换算成固定持续秒数。

每模型三级前两轮各10次左右目全部保存，实际各120个MP4（3级×2轮×10次×2目）。第3–5轮不录视频，继续记录统计和命令。先前材质/Kit启动技术中止的输出另目录保留，未混入当前成绩。

## 场景材质与观察到的异常

烤箱旧包对应HSSD场景102344280，原USD SHA256 `e0f131f5c3ae8057cd8afd11c7b4382f564e7b53236df1f079f6b229cbe8a58e`。场景文件有79个失效外部引用路径。包装器在官方Kit初始化后、场景加载前，仅进程内把失效props引用重定向到该场景原包中的同名原文件，持有Layer；不Save、不写官方USD、不造替代材质、不把其他场景同名props混入。该场景reference位置计数180保持；79是重定向的唯一外部路径数，不是材质总数。

当前30份烤箱console.log（两模型×三级×五轮，含进行中日志）扫描：指定缺失USD/解析失败/UsdToMdl/纹理加载失败模式0，Python Traceback0，collision mesh无points模式0。仍有各模型69条Kit `[Error]`标记，抽查对应torchvision视频编码/解码deprecated UserWarning经stderr记录；不能将标签数直接当渲染故障数，也不把关键词扫描当全材质验收。

level0/repeat1/episode0视频人工检查：STATE接近把手后基本停滞至时限；VGQ-AE继续拉开门并成功。两份同初态视频均见前下方边缘整齐黑块；未定位具体prim/材质/阴影，尚无去除黑块的配对因果实验，不能称已修复或认定它解释STATE差距。此前π0.5两桌换块回撤诊断不是本次GR00T烤箱根因证据。

## 硬件、环境与并行分配

8×NVIDIA RTX5090约32GiB，驱动580.159.04。GR00T Python3.10.12、Torch2.7.0+cu128、Transformers4.51.3、FlashAttention2.7.4.post1、NumPy1.26.4、Albumentations1.4.18、TorchCodec0.4.0；仿真使用独立环境Isaac Sim4.5.0.0、MuJoCo3.3.6。

烤箱STATE L0/1/2分别GPU0/1/2、端口8120/8121/8122、DDS100/101/102；VGQ-AE对应GPU3/4/5、端口8123/8124/8125、DDS103/104/105。每实例模型和仿真同卡，每进程8CPU线程，Kit单GPU渲染；DDS loopback接口lo。各实例独立服务，不并发共享一个模型对象。前向预热实际返回有限(16,36)后才启动仿真。后来抱箱另用GPU6/7，与本次任务不同。

## 实际启动命令：STATE level0第1轮

以下据保存的worker.json/command.sh整理为相同参数的易读格式。端口及输出路径属于当时运行；正在运行时不要再次执行造成冲突。其它难度/变体替换上表对应GPU、端口、DDS、权重和输出路径；每级五轮由既有worker调度，后3轮关闭录像。

模型服务：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh gr00t
cd /root/workspace/RHSCI-Private
CUDA_VISIBLE_DEVICES=0 python -m rhsci.inference.http_server \
  --backbone gr00t \
  --run-dir "$PSI_RUNTIME_ROOT/checkpoints/followup/gr00t-STATE-oven/inference" \
  --checkpoint-step 20000 --device cuda:0 \
  --host 127.0.0.1 --port 8120 \
  --no-deterministic-eval
```

客户端（实验包装器调用官方simple.cli.eval_decoupled_wbc.run_eval，并非另写仿真/控制框架）：

```bash
source /root/workspace/Haoxuan/RHSCI-runtime/activate.sh simple
cd "$PSI_RUNTIME_ROOT/evals/gr00t-official-pre1102-5cycles-20261104/gr00t-STATE-oven/STATE/dr-level-0/repeat-01"
CUDA_VISIBLE_DEVICES=0 python -u \
  /root/workspace/RHSCI-Private/experiments/pi05_between/eval_decoupled_wbc.py \
  simple/G1WholebodyOpenOvenTeleop-v0 psi0_decoupled_wbc train \
  --data-format lerobot --sim-mode mujoco_isaac \
  --data-dir "$PSI_RUNTIME_ROOT/eval-official-pre1102/G1WholebodyOpenOvenTeleop-v0/dr-level-0" \
  --host 127.0.0.1 --port 8120 --headless \
  --num-episodes 10 --episode-start 0 \
  --max-episode-steps 1000 --success-criteria 0.5 \
  --eval-dir "$PWD/stats" --save-video \
  --evaluation-round 1 --video-rounds 2 \
  --renderer-gpu 0 --dds-domain-id 100 --cpu-threads 8 \
  --repair-scene-references
```

## 原始证据位置

- 结果根：`/opt/rsci-data/evals/gr00t-official-pre1102-5cycles-20261104/`，两模型目录分别gr00t-STATE-oven、gr00t-VGQ-AE-oven。
- 每级worker.json：GPU/进程/端口/DDS及实际服务参数；warmup.json：真实返回形状；COMPLETE.json仅完整50次后存在。
- 每轮effective_eval_config.json、command.sh、console.log、scene-reference-repair.json、stats/eval_stats.txt（具体stats文件可在stats内查找）、verified_results.json。
- 权重目录：`/opt/rsci-data/checkpoints/followup/gr00t-{STATE,VGQ-AE}-oven/`，包含checkpoint-20000/{config.json,processor_config.json,statistics.json,launch_contract.json}及训练源码快照/日志。
- 旧包收据：`/opt/rsci-data/eval-official-pre1102/gr00t-old-package-source.json`。
- episode0视频：结果根下`gr00t-STATE-oven/STATE/dr-level-0/repeat-01/data/evals/psi0_decoupled_wbc/G1WholebodyOpenOvenTeleop-v0/dr-level-0/episode_0/head_stereo_left_failed.mp4`；VGQ-AE对应路径变体名VGQ-AE、文件名head_stereo_left_success.mp4，右目文件也保留。

## 官方来源

- [SIMPLE论文，附录A.3与A.5表8](https://arxiv.org/html/2606.08278v2)：发布权重、20k/全局24、顺序无RTC，以及16任务联合评测结果；未逐项明确我们以上全部推理参数。
- [固定官方GR00T README](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/baselines/gr00t-n1.6/README.md)。
- [固定G1 modality配置](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/src/gr00t/gr00t/configs/modality/g1_locomanip.py)：有效16步、10组ABSOLUTE。
- [固定训练预设](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/baselines/gr00t-n1.6/presets/train/finetune_simple.yaml)：本地初始化路径，仅作公开配置证据。
- [本次SIMPLE源码版本](https://github.com/physical-superintelligence-lab/SIMPLE/tree/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11)。
