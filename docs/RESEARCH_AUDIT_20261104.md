# 整理版科研审查：代码、数据、复现和评测

本轮以 RHSCI-Private/main `d0ac783` 为修改起点，沿用现有框架，只修复已复现的 GR00T 检查点发现问题。报告文件按本会话日期命名，JSON 中的 `date_utc` 为机器实际采集时间。PSI0-RAW 生产代码、活动进程、数据原件和检查点均未改动；未启动正式训练或完整仿真评测，未推送远端。

**结论：五套现用数据的数值、标签和统计配对可靠，五个骨干的已执行 CPU 检查通过；仍不能把当前项目称为“全模型、全部数据和物理评测均已严格复现”。** 主要未闭合项为演示成功性验收、独立验证集、λ₀ 全模型环境、多卡长程与物理闭环，以及官方四任务成功判定的配套缺口。

## 本轮修复

`rhsci.inference.checkpoints.inspect_run()` 原先对 π0.5、GR00T 都要求 `rsci_config.json`。真实 GR00T 原生检查点保存的是 `config.json`，导致公共 HTTP 的 `--discover` / `--run-dir` 报“没有完整模型”，直接调用原生服务器则不经过此处。

已按骨干选择原生配置文件名。新增回归先复现失败，再验证纯推理分片导出能发现、缺片仍拒绝；不增加训练状态依赖。六份真实 GR00T 20k 检查点现均通过发现入口，见[真实检查点收据](GR00T_DISCOVERY_AUDIT_20261104.json)。这项核验读取配置、分片清单及大小，不表示已完整加载六个大模型。

另外追查了公共参数转发对显式 False 的疑点：EgoVLA 实际转发原始 `sys.argv`，π/GR 现有参数也没有触发该分支的可关闭默认 True 开关。因此没有将其记为现行训练缺陷，未作无依据修改。

## 全量训练数据核验

本机数据根：`/mnt/cfs/cqizhh/personal/huangkailun/haoxuan/data`。精确路径、数据指纹、标签清单 SHA、教师版本、所有统计差异和缓存身份保存在[全量数据报告](TRAINING_DATA_AUDIT_20261104.json)。

| 数据 | 轨迹 | 帧 | 有效 ARDY 帧 | 不补尾的16步窗口 | 不补尾的30步窗口 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 两桌 | 91 | 58,637 | 58,091 | 57,272 | 55,998 |
| 水龙头 | 99 | 39,368 | 38,774 | 37,883 | 36,497 |
| 递物 | 100 | 47,259 | 46,659 | 45,759 | 44,359 |
| 抱箱 | 94 | 47,772 | 47,208 | 46,362 | 45,046 |
| 烤箱 | 99 | 37,903 | 37,309 | 36,418 | 35,032 |
| 合计 | 483 | 230,939 | 228,041 | 223,694 | 216,932 |

逐轨迹检查覆盖：50 Hz、36维动作/32维状态、episode/frame/global index、时间戳、有限值、任务索引、视频存在及容器声明帧数；全部 ARDY NPZ 的 SHA、形状、掩码和 processed fingerprint；每集前6帧不计 ARDY 损失。五套原始帧索引增量均为1。`stats.json` 与 `stats_psi0.json` 的 min/max/mean/std/q01/q99 由实际全部训练帧重算，均在 `rtol=1e-6, atol=1e-7` 内匹配；没有发现越出保存 min/max 的帧。

全部483个视频容器为 H.264、640×360、50 Hz，容器声明帧数与表一致。容器元数据核验不覆盖每个像素的解码正确性。

两桌、抱箱共享原始缓存共185条轨迹、372个载荷文件、73,583,685,930 bytes：全部载荷及源视频 SHA 匹配；缓存动作/状态与 Parquet 精确一致，全部 PTS 正确。另独立重解码每集20次查询，覆盖尾/首/历史边界/重复/固定随机帧，共3,700次查询，像素与 PTS 精确一致。五套历史包装清单中的1,006个文件也全部按大小、SHA复核。见[来源和像素补充证据](DATA_PROVENANCE_RGB_AUDIT_20261104.json)。这不是对230,939帧全部RGB逐像素重解码。

### 必须披露的数据问题

1. **递物100条不等于100条独立运动示范。** 动作与状态整条逐字节相同的7对为 `(13,93)、(22,94)、(26,95)、(31,96)、(35,97)、(39,98)、(43,99)`，和原 `provenance/package.json` 的 `duplicate_raw_groups`、`unique_raw_sources=93` 一致。它们可能有不同视觉渲染，不能据运动数组相同就声称所有图像相同。训练会重复加权这些运动，按轨迹划分验证时应按原始来源分组；本轮未删除、重排数据或改变既有配方。
2. **格式验收与成功演示验收是两件事。** 五套包装记录仍为 `status=format_complete, visual_contact_acceptance=pending, training_enabled=false`；递物还明确记载 `source_success_certified=false`，其后续更新记录也保留 pending。这些是历史包装字段，用户之后的训练授权/实际运行以当前记录为准，不能把旧字段变成新的启动禁令。但本轮没有找到覆盖当前全部轨迹的新增视觉接触/物理成功验收记录，也没有重放演示，不能宣称“483条都已验为成功”。
3. **稀少变化被 v2 量化归一化抹掉的旧问题仍在。** 两桌动作0–6维、烤箱动作31维及状态31维有分位数近等但实际变化的情况。统计正确不等于该归一化保留了全部信息。按用户此前“只修恢复”的明确要求保留 π0.5 v2；不更换旧模型统计，也不以本轮检查暗示已修。计数受“偏离分位数超过多少”的阈值影响，详见 JSON 中明确的 `1e-6` 判定和[历史说明](PI05_CORRECTNESS_20261102.md)。
4. **XMovePick 当前本地恢复不完整。** `simple-render-fixed-h264-20261014/G1WholebodyXMovePickTeleop-v0` 有 episodes/provenance/validation，另有 ARDY 旁路，但该目录没有当前入口需要的 `meta/info.json`。它未计入上述五套；有备份或标签目录不表示当前可直接训练。需要按[资产清单](ASSETS.md)恢复匹配的处理后包，再核验，不自动转换其他布局。
5. **教师核验范围有限。** 本轮重验全部标签文件及配对身份；教师记录固定 ARDY 源码、权重 revision，并引用 XMovePick 单条 golden reference。没有重新运行教师为五套数据全量再生成标签。`formal_verified` 字段与完整文件 SHA 不能替代新的全量教师数值复算或演示成功性验收。

## 实际入口、划分和公平比较

已复用生产预检执行56组真实输入检查，无全尺寸模型更新：

| 骨干 | 真实输入检查范围 | 证据 |
| --- | --- | --- |
| Ψ0 | 两桌/抱箱 × STATE、NO-STATE、VGQ-AE、VGQ-VLM、STATE-RECONSTRUCTION；含实际本地处理器/AE资产存在性及预处理/拼批 | [10组](ADAPTER_INPUT_PSI0_20261104.json) |
| ACT | 两桌/抱箱 × STATE、NO-STATE、EGQ-AE；全表归一化与首/尾/跨集边界 | [6组](ADAPTER_INPUT_ACT_20261104.json) |
| π0.5 | 五任务 × STATE、NO-STATE、VGQ-AE；有效动作窗口、历史和状态依赖 | [15组](ADAPTER_INPUT_PI05_20261104.json) |
| GR00T | 五任务 × STATE、NO-STATE、VGQ-AE；有效动作窗口、历史及标签/状态依赖 | [15组](ADAPTER_INPUT_GR00T_20261104.json) |
| EgoVLA | 五任务 × STATE、VGQ-AE；30步原始动作、历史/标签及状态依赖 | [10组](ADAPTER_INPUT_EGOVLA_20261104.json) |

Ψ/ACT另三任务未做本轮生产入口预检。ACT首次读取水龙头会经原生LeRobot生成包含未用深度列的大型HF Arrow缓存；已停止**本轮自己的审计进程**，未触及正式任务，未保留其大载荷。后续用现有CFS HF缓存完成两桌/抱箱检查。原生读取会写外部缓存，和只读的 Parquet/标签/统计审计区分。新机将 `HF_HOME` / `HF_DATASETS_CACHE` / `TORCH_HOME` 放容量足够的外部盘；源数据压缩包大小不能当成运行缓存占用上限。本轮未为完成检查而重写加载框架。

**当前没有自动建立独立留出集。** 五套 metadata 都只声明 train。Ψ0 `LerobotDataConfig.check_repo_ids()` 默认把第一个训练 repo 作为 val；`LeRobotDatasetWrapper` 不按 metadata splits 再筛独立轨迹。真实 VGQ-VLM 的保存配置也确认 train/val repo 相同。ACT worker 创建同一数据的前缀 Subset 作为 eval，并保留原生训练增强。两者日志中的验证损失只能描述训练分布的诊断指标；π/GR/Ego 当前 RHSCI worker没有另接独立 eval_dataset。

保持既有复现配方，不临时切走一部分历史训练帧。新研究若需离线泛化指标，应在新实验中按原始演示来源分组划 train/val，重复来源不跨组；归一化只从新 train 算，相关标签/缓存按实际变化核对，使用独立运行目录。不能把相邻帧随机拆分当作独立轨迹验证，也不能用 SIMPLE 测试成功率反复选择超参数后仍称未见测试集。

不同骨干的样本基数不同：Ψ/ACT保留尾帧并mask动作，π/Ego只取完整30步，GR只取完整16步但内部补到50。ARDY前6帧被mask不代表对应动作样本全部删除。跨方法应固定同一骨干的数据/初始化/预算/seed；跨骨干另报告实际消费样本数、全局batch、更新数及预训练来源。GR当前默认NVIDIA N1.6初始化和SIMPLE固定50k初始化仍有差异。状态重建的31维监督保留原ARDY有效窗口掩码，ARDY数值不参与其重建损失；不能把它描述成完全无sidecar依赖的入口。

## 模型、恢复和测试证据

本轮CPU回归：Ψ0 62、ACT 30、EgoVLA 29；修复并把公共发现测试接入默认runner后，π0.5 44、GR00T 40；公共HTTP12、训练CLI/结构15；公共契约/完整状态/评测资产等17项实际执行、2项官方集成最初跳过，随后设置真实 `SIMPLE_HOME` 将这2项单独执行通过。λ₀守护/来源9项通过。不要将重跑和交叉收录的测试相加为独立覆盖率。

λ₀完整模型导入在现有Ψ环境明确失败：Transformers缺少 `Qwen3_5Config`；本机没有本轮可用的λ独立模型环境，因此没有全模型、bundle和数据处理验收。未用假模型或其他骨干测试代替。CPU runner按现行规则排除了分布式测试；多卡更新、worker内部随机增强、完整长程恢复、GPU推理及物理闭环均未新增验收。

初始化大权重和所有生产模型载荷本轮没有全部逐文件重新哈希；本轮新增的真实检查点证据限于GR发现入口的配置/分片清单/大小，Ψ预检只检查所需本地资产及实际处理器。不能用小模型回归、文件头或下载清单的摘要替代完整载荷验收。

沿用[原/整理版配对](RAW_COMPATIBILITY_AUDIT_20261103.md)的13组微型三次更新、20组预处理对照、真实ACT权重推理和其他骨干生产权重文件头证据。这些是此前执行，不重复声称本轮重做。对应原模型兼容不意味着默认新训练命令与历史命令相同，也不意味着两版完整训练/成功率一致。恢复修复、BF16执行配置等差异必须按实际记录解释。

任务开始已有未提交的 `src/egovla/VILA/llava/model/multimodal_encoder/builder.py` 改动（Intern视觉塔改为分支内延迟导入）。本轮保留且不提交它；审计收据记录其实际SHA。因此当前工作区身份应表示为本地提交加该已有改动，而非干净HEAD。尚未核验生产尺寸Ego初始化，不能把微型测试解释为这一实际路径完整通过。

源码交接检查还发现：`docs/DIRECTORY_STRUCTURE.md` 与HEAD一致，但旧公开清单中的SHA过期，`experiments/README.md` 也漏列，本轮纠正这两处清单记录；工作区另有31个本轮开始前生成的忽略字节码文件。原有编辑及字节码均未改动、未纳入提交。完整工作区扫描会如实报告它们；交付检查针对暂存源码导出另行执行，不能以忽略文件掩盖为工作区全清洁。

## 官方、评测数据和报告口径

本轮重新GET核对九个公开端点：[官方元数据](UPSTREAM_METADATA_20261104.json)。SIMPLE main仍为 `b75c6786`，Psi0为 `cda3fdfa`；新独立评测库 `fb67cd9a`、修正源 `ae76945d`、psi-data `de140d67`、合并训练集 `4710ec03`、模型 `76febdc9` 与前次记录相同。这个结论限定查询的主分支/发布身份，不表示扫描了所有组织、未合并分支或第三方镜像。

本地SIMPLE的433个顶层文件和9个递归子模块再次匹配固定版本；`sync_simple_eval.py --verify-only` 复验独立评测库2,993文件，以及25份ZIP的完整摘要/CRC/解包字节。物理场景大资产没有全量重新下载或逐文件重新哈希，模拟器及控制器环境未做完整安装/运行验收。

继续保留[前次确认的差异](SIMPLE_UPDATES_20261103.md)：BendHandover、BendPickAndPlace、BendPickAndPlaceOnSofaMP、PickAndBendPlaceMP四任务的新成功规则未出现在官方固定代码，两桌/抱箱不属于这四项；历史WBC手指接口问题未随数据更新消失。本轮不凭README猜改奖励，也不改用户此前搁置的控制器接口。

正式结果记录代码/子模块、权重及处理器、训练数据/统计、评测资产/场景、任务/指令、等级/episode列表、seed、控制频率、执行horizon、推理步数、RTC/噪声协议及成功判定。报告成功数/总数、技术失败/重试和录像，保留失败attempt；HTTP200或退出码0不证明episode完成。控制频率与训练50Hz时间轴应按协议一致映射，不把推理请求次数当成历史帧时间。

同权重重复回放同一初态/随机流不增加独立样本量。分别报告训练seed间差异和固定episode上的评测波动；比较方法用相同episode/seed成对比较，置信区间说明独立单元与重复结构，不能把相关帧或同轨迹重复当独立样本。旧资产、旧success checker及不同初始化协议的成绩单独列示。新训练包有删轨迹、动作重定向和补帧时，按帧身份/受影响字段决定重建ARDY、缓存及统计；旧权重保留训练时统计。

## 复核命令

全量文件级审计独立于正常训练入口，不增加启动门禁；五套精确配对见 JSON，也可一次传多组：

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/psi-env/bin/python -B tools/audit_training_data.py \
  --pair /data/processed/TASK /data/formal-ardy/TASK \
  --cache-root /data/.cache/act/space-cache-v1 \
  --probe-videos --hash-cache --verify-provenance --compare-cache-rgb \
  --report /outside-checkout/training-data-audit.json

PYTHONDONTWRITEBYTECODE=1 /path/to/backbone-env/bin/python -B tools/audit_adapter_inputs.py \
  --backbone pi05 --data-report /outside-checkout/training-data-audit.json \
  --report /outside-checkout/pi05-input-audit.json

python3 tools/sync_simple_eval.py --output-root /data/SIMPLE-eval-20261103 \
  --release snapshot-fb67cd9a --verify-only
python3 tools/sync_simple_eval.py --output-root /data/SIMPLE-eval-20261103 \
  --release zip-38e43f22 --verify-only
```

真实输入工具也接受gr00t、egovla、act、psi0及重复的 `--task` 精确筛选；Ψ0还需显式 `--vlm-path` / `--ae-path`。原生预检可能生成外部HF缓存；先配置容量足够的 `HF_DATASETS_CACHE`，勿将其指向仓库或当前活动任务的临时目录。采集范围、通过/未验项和测试环境汇总见[审计收据](RESEARCH_AUDIT_20261104.json)。
