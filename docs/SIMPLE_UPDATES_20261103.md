# SIMPLE 代码与资源更新核验（2026-11-03）

所属 RHSCI-Private/main。更新 SIMPLE、下载评测资源或解释“官方是否更新”前必读。本次只在本地提交，不推送。代码、场景资产、训练数据、评测数据和权重分别固定版本。

## 搜索遗漏的更正

11 月 1 日实际比较的两份旧包确实逐文件相同，但搜索漏查了 `SIMPLE-org/data/eval-fixed/`。该目录的两桌修正包在 **10 月 30 日 18:26:42 UTC** 已提交，早于当时核验；不能以 `USC-PSI-Lab/psi-data` 当时没更新推断修正未公开。旧[核验记录](SIMPLE_OFFICIAL_20261030.md#eval-layout-audit-20261101)与 JSON 保留为历史证据，本记录覆盖其搜索范围结论。

本次沿官方 README、下载脚本、HF 提交来源和两个发布组织的公开仓库列表查找，并检查 SIMPLE/Psi0 的 main、公开分支及 SIMPLE 最近 PR。发现修正源目录、同步目录和新独立仓库之间也存在版本差异。

## 当前发布及处理

| 部分 | 本次固定版本 | 结果 |
| --- | --- | --- |
| [SIMPLE 代码](https://github.com/physical-superintelligence-lab/SIMPLE) | `b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11` | main 未变；保持已核验的 433 个顶层文件与 9 个递归子模块版本 |
| [Psi0 代码](https://github.com/physical-superintelligence-lab/Psi0) | `cda3fdfafe47cb6fa75400d013de1bd36e3cea05` | main 未变；两仓库另有 website 分支，不能当训练/评测代码更新 |
| [SIMPLE 场景/机器人资产](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE/tree/0e8f23cce8f2e96717aa60a0e26bed9d56223c2d) | `0e8f23cce8f2e96717aa60a0e26bed9d56223c2d` | 与上次记录相同，未重复下载大场景资产 |
| [修正源目录](https://huggingface.co/datasets/SIMPLE-org/data/tree/ae76945d457e9e55a4ba196c2a18b911e8889547/eval-fixed) | `ae76945d457e9e55a4ba196c2a18b911e8889547` | 17 个修正 ZIP 与下行发布包的大小、LFS SHA256 全部相同 |
| [psi-data/simple-eval](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/38e43f22c70390f9cb813d8bb047395398c2a043/simple-eval) | 同步提交 `38e43f22c70390f9cb813d8bb047395398c2a043` | 已下载、解包并复验全部 25 ZIP；其中 17 个来自修正源，8 个为其他既有任务 |
| [独立 SIMPLE-eval](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE-eval/tree/fb67cd9a309cb4f3ce8f9470fecb6fa6d82c60d4) | `fb67cd9a309cb4f3ce8f9470fecb6fa6d82c60d4` | 已同步 17 任务、2993 文件；包含 11 月 3 日语言修正，作为当前示例入口 |
| [psi-data 训练数据](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/de140d67ba77a541f2a58354f7edf4e2480b7c56/simple) | `de140d67ba77a541f2a58354f7edf4e2480b7c56` | 已核元数据，17 个源 `data-fixed` 包全部匹配；未覆盖现有训练数据 |
| [合并训练集](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE-decoupled-wbc/tree/4710ec03bd726d02dd034df0f951cc6df6b9ed35) | `4710ec03bd726d02dd034df0f951cc6df6b9ed35` | 新发现的独立发布入口；已读 README 和固定身份，未下载训练载荷 |
| [psi-model](https://huggingface.co/USC-PSI-Lab/psi-model/tree/76febdc998e5d32a1ed978c286018752676fce64) | `76febdc998e5d32a1ed978c286018752676fce64` | 有新微调权重及撤回；下述现用初始化文件身份未变，未下载大权重 |

[上游审计 JSON](../configs/simple/upstream_audit_20261103.json)保存查询地址、固定提交、提交说明、训练包身份及初始化对照；[评测清单](../configs/simple/eval_assets.json)保存全部下载文件的大小与 HF LFS SHA256 / 普通 Git blob SHA1。普通 Git blob 摘要含 `blob <size>\0` 头，不能直接按文件 SHA1 验证。

SIMPLE [PR #31](https://github.com/physical-superintelligence-lab/SIMPLE/pull/31)涉及 renderer 清理，检查时仍未合并。本次没有把未合并 PR 或独立子模块最新 main 混入官方固定版本。

## 两桌修正的实际内容

新版 ZIP 为 **689559 bytes**，SHA256 `7d215cf3e9202b3929d80254441553d18089c582f81376c17001c11eecc28dee`；旧版为 480532 bytes，SHA256 `8a4b0ff7d068967dac8baec2c4a4deca4f644215a1ee15e921739f3dfe90e7e0`。

实际文件从 75 增至 104：共同路径中 66 个内容变化，新增 29 个，未删除旧路径。新增 `dr-level-hard`。原 level-0/1/2 的 30 条机器人出生位姿全部变化；目标位姿仅 level-2 的 10 条变化。PR 指定的五条目标位置如下（米，显示六位）：

| level-2 episode | 旧 x / y | 新 x / y | z |
| --- | --- | --- | --- |
| 0 | -0.314270 / 0.058306 | -0.311697 / -0.008627 | 0.763422 |
| 6 | -0.330995 / -0.051027 | -0.325030 / -0.011364 | 0.763422 |
| 7 | -0.329937 / 0.013423 | -0.315339 / 0.017011 | 0.763422 |
| 8 | -0.305153 / -0.051186 | -0.325750 / 0.036746 | 0.763422 |
| 9 | -0.329676 / 0.017821 | -0.323974 / 0.035257 | 0.763422 |

来源为 `meta/episodes.jsonl` 中的 `environment_config.dr_state_dict.spatial["0"]`，机器人为 `spatial.g1_sonic`。还有干扰物、材质等配置变化，不能把新版称为只平移五个物体的补丁。[逐 episode 证据](../configs/simple/eval_layout_audit_20261103.json)保留全部 30 条新旧目标/机器人位姿与文件差异。已核实资产变化，未验证初始化碰撞、物理稳定性或成功率。

## 其他容易漏掉的差异

**ZIP 不是最新独立评测库的完全替代。** 对最新 17 任务逐文件比较：2978 文件与 ZIP 解包内容相同，13 个内容变化，另外增加仓库级 README 和 `.gitattributes`。变化文件集中在水龙头、XMovePick 的四难度 metadata：水龙头指令从 `move forward to the faucet and open it` 改为 `turn the faucet`，XMovePick 从 `bend to pick up the {}` 改为 `move forward to pick up the {}`。这是 11 月 3 日的新提交；不能只按 11 月 2 日 ZIP 同步认定所有资源已更新。

**四任务成功判定与公开代码不配套。** 新评测库 [README](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE-eval/blob/fb67cd9a309cb4f3ce8f9470fecb6fa6d82c60d4/README.md)描述以下变化；实际检查固定 `b75c6786` 对应函数后确认尚未包含：

| 任务 | 新发布说明 | 当前公开代码差异 |
| --- | --- | --- |
| BendHandoverTeleop | 箱子质心须在篮子 xy 范围内 | `check_object_in_container()` 仅判断接触 |
| BendPickAndPlaceTeleop | 同上 | `check_object_in_container()` 仅判断接触 |
| BendPickAndPlaceOnSofaMP | 放到沙发并松手，奖励舍入避免 0.1 累加误差 | `compute_reward()` 缺少松手门槛和舍入 |
| PickAndBendPlaceMP | scene19 狗窝接触且不能接触地面 | `check_object_in_container()` 仍以 `container_name = "world"` 判断 |

因此这四任务不能以“新资产已下载”声称新版官方评测逻辑已严格复现。两桌、抱箱不在这四项中。保留官方源码，不凭 README 猜写奖励替代作者实现。新训练集还引用 SIMPLE `c5fb4bd` 的视频脚本，但该短提交无法在已核官方仓库 API 定位；不能把它标成已合并的公开代码。

**训练数据不只是重新压缩。** 新[训练集说明](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE-decoupled-wbc/blob/4710ec03bd726d02dd034df0f951cc6df6b9ed35/README.md)记录：5 个 MP 任务手部动作重定向、失败轨迹剔除；合并集为 1517 轨迹 / 606329 帧；600 条轨迹有尾部保持补帧，补帧上限现为 50。psi-data 另有三任务统计重算、视频 GOP10/无 B 帧提交。上述内容来自作者说明和固定元数据，未逐帧重验大训练包。采用新版时，应按实际变化核对历史 ARDY、缓存和归一化的对应关系；不能只凭任务名相同就复用，也不能一概宣布全部旧标签失效。现有 RHSCI 训练保持原保存版本。

**新权重中也有撤回。** `psi-model` 新增 mix24/mix33、BendPickPlace、Handover/XMovePick 和 decoupled17 等权重，但最新提交删除了 `dwbc17...2611020337`；不能把上传历史当当前可用清单。当前 `decoupled17...2610310854` 的 20k/30k 模型仍存在，每份 11219871480 bytes，目录只有模型文件，不能称完整可续训点。现用 Ψ0 VLM（12 文件）、AE（1 文件）和 `openpi/pi05_droid`（2 文件）的固定文件身份与旧版一致。`SIMPLE-org/g1-tabletop-grasp` 是另一已发现模型入口，最后更新为 2026-01-29，没有作为本次新权重使用。

官方资产脚本和 `resolve_data_path()` 使用浮动 main，后者已有目录时不会自动刷新。这次以固定版本、独立目录和显式核验解决评测资源更新，未改第三方脚本，也没有将“目录存在”当版本正确的证据。历史 `ds1x/RHSCI` 资产交接不是这批新评测包。

## 历史手指映射问题仍未随官方更新解决

2026-11-03 用户追问后复核：当前官方 main 仍为 `b75c6786`。历史发现的 **WBC / MuJoCo 食指与中指顺序不一致**仍在当前 RHSCI 原生官方评测链路中，不能把本次资源同步当作它已修复。

- `psi0_decoupled_wbc.py::get_action()` 把 WBC `get_hand_actuated_joints()` 返回值直接装入左右 `hand_q`；WBC supplemental info 的顺序为 thumb/index/middle。
- `g1_sonic.py` 按 MJCF 遍历建立 `left_hand_index/right_hand_index`，`apply_action()` 直接按该数组做手部 PD，反馈 `prepare_obs()` 也按这个数组取值。历史实际 XML 的顺序为 thumb/middle/index，接口未按具名关节转换。
- 当前固定资产库的 `robots_g1_sonic.zip` 为 23114187 bytes，HF LFS SHA256 `6c802053e1777f675068795d6100dd9c2b1e1853252ad31c0bf3c4285ef76b63`，与历史诊断记录完全相同。此处核发布身份，没有重新下载并重算整包摘要。

[原生机器人代码](https://github.com/physical-superintelligence-lab/SIMPLE/blob/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11/src/simple/robots/g1_sonic.py)及当前锁定的 WBC 子模块已直接读取。历史真实 XML / PD 测试证明食指目标可驱动另一手指；本次复核源码路径和相同资产身份，没有重跑物理或成功率测试。

WBC 出处更正：前次给用户的链接指向 GR00T-WholeBodyControl 内的副本。SIMPLE 的 `pyproject.toml:134` 和 `uv.lock` 实际指定 `third_party/decoupled_wbc`，固定 `songlin/decoupled_wbc@5b304e4d643fe89cd000068dcbba7e7519539caa`。已逐字节比较两份 `robot_model.py`，内容相同；下面用实际依赖的链接补齐调用链，不能仅凭副本路径推断运行来源。

1. [G1 名称清单第81行](https://github.com/songlin/decoupled_wbc/blob/5b304e4d643fe89cd000068dcbba7e7519539caa/control/robot_model/supplemental_info/g1/g1_supplemental_info.py#L81)按 thumb/index/middle 排列，第319行原样传给基类 dataclass。`instantiation/g1.py:53–60` 把此实例交给 `RobotModel`，由 SIMPLE `SonicDecoupledWbcAgent._init_decoupled_policy()` 调用。
2. [RobotModel 构造函数第52行](https://github.com/songlin/decoupled_wbc/blob/5b304e4d643fe89cd000068dcbba7e7519539caa/control/robot_model/robot_model.py#L52)逐项遍历清单，调用 `dof_index(name)` 生成 `_left_hand_actuated_joint_indices`；不排序、不去重，右手同理。
3. [dof_index 第132行](https://github.com/songlin/decoupled_wbc/blob/5b304e4d643fe89cd000068dcbba7e7519539caa/control/robot_model/robot_model.py#L132)仅按关节名查 `joint_to_dof_index`。该字典由构造函数第32–35行按 Pinocchio `jmodel.idx_q` 建立；查询不改变遍历清单的顺序。
4. [索引 getter 第168行](https://github.com/songlin/decoupled_wbc/blob/5b304e4d643fe89cd000068dcbba7e7519539caa/control/robot_model/robot_model.py#L168)在 `side="left"` 时直接返回上述缓存数组，右手返回另一缓存数组。
5. [目标 getter 第336行](https://github.com/songlin/decoupled_wbc/blob/5b304e4d643fe89cd000068dcbba7e7519539caa/control/robot_model/robot_model.py#L336)用该索引数组取 `q` 的对应元素。NumPy 按索引数组顺序输出，因此结果保持 thumb/index/middle，而非把索引数值排序后输出。

旁边 `get_joint_group_indices()` 的 `sorted()` 用于另一条关节组接口，不参与上述手指索引 getter。这里证明 WBC 输出数组顺序；整体错配还须结合前述实际 MJCF 顺序、客户端直接传递及 PD 路径。当前独立环境缺 Pinocchio/MuJoCo，因此本次为固定源码调用链复核，不伪称新运行了完整 WBC/物理链路。

PSI0-RAW 历史本地 `scripts/eval_wbc_hand_order.py` 的 `wbc-hand-names-legacy-actions-v2` 是额外适配，处理目标/反馈具名映射及旧 checkpoint 动作兼容。用户要求 RHSCI 完全替换为官方 SIMPLE 时，该本地补丁没有迁入；当前 RHSCI 中也未找到它的安装调用。因此不能用“旧项目曾修复”作为当前入口已经修复的证据。

这个控制接口问题与新版 **MP 训练数据 soft Dex3 手部动作重定向**不同。新训练包、语言或场景位置修正不会自动修复接口。它可能影响两桌、抱箱等使用该链路的任务；这两任务不属于前述四项成功判定更新，不代表它们没有其他已知问题。历史核验也没有证明本机训练转换新增了手指标签错位，不能据接口问题直接要求重训或交换训练标签。此次只补充缺陷记录，未改官方源码、训练数据或活动进程。

## 对当前实验的影响与遗漏教训

25 份 ZIP 和独立仓库的 17 个任务是有重叠的两种发布，不是新增 42 个任务。前者保留发布包供对照，后者作为当前评测示例入口；17 个任务中大部分内容相同，两任务的指令又有后续修正。评测包指定开局环境、对象/机器人位姿和语言等输入；任务是否成功由仿真任务代码里的函数判断。下载新包不会自动改写这四任务的成功判定函数，也不等于模型已重新训练或完成新版评测。

训练数据变化对派生产物的影响须分别判断：

| 变化 | 对既有 ARDY / 缓存 / 统计的影响 |
| --- | --- |
| 手部动作重定向 | 动作缓存须与新目标一致；依赖被修改字段的标签需重算，只依赖未变身体运动的标签不因此自动失效 |
| 删除轨迹、重排 episode 编号 | 标签和缓存要按实际轨迹重新对应，不能直接沿用旧编号 |
| 增删尾部保持帧 | 时间索引、长度和尾部标签须匹配新版；不能用旧帧表代替新帧表 |
| 更新归一化统计 | 新训练按所选数据与原生配方配套统计；旧 checkpoint 推理仍使用其训练时统计，不自动换成新统计 |
| 仅视频重新编码、帧序和运动数据不变 | 不能据此要求所有运动标签重算；RGB 像素可能变化，图像缓存须核对所用版本 |

因此，当前训练继续用原数据、原 ARDY、原缓存和原统计，并不因上游更新而自动作废。本次仅更新独立评测资源和使用示例；未将新训练包混入正在运行的实验。迁移训练数据时再处理具体受影响的产物，不据本记录要求全部重训。

这次遗漏的根因是：把已知镜像入口当作公开来源全集；发现旧包相同时没有继续追查 PR 所指的修正源；把有限范围内正确的字节比较外推成“未公开修正”。镜像在 11 月 2 日更新，不能掩盖修正源 10 月 30 日已存在这一事实。

以后检查官方更新须完成以下工作，再作结论：

1. 分开检查代码、评测数据、训练数据、场景资产和权重，记录各自固定版本。
2. 沿 README、下载脚本、PR/提交说明追查源目录，同时查看发布组织的新独立仓库；SIMPLE 至少覆盖本记录的修正源、镜像和独立发布入口。
3. 对照源提交时间和镜像同步时间，并核文件身份及关键内容。遇到明确提及但未找到的修正，继续追来源；无法找到则写明缺口。
4. 区分“文件相同”“新资源已下载”“代码已实现”“物理验证通过”，逐项报告证据，不互相代替。
5. “无更新”只能限定为已核仓库、分支和目录；网络失败或来源缺失写“未完成核验”，不能写成“没有更新”。

这是本次已落实的调查要求，不增加训练/推理启动门槛。后续查更新前先读本节和主树错误账本。

## 下载和本机落地

在 RHSCI 检出目录执行，任意标准 Python 3.10+ 即可，不需安装 HF 客户端：

```bash
# 最新 17 任务；只下载一个任务时添加 --task <完整任务名>。
python3 tools/sync_simple_eval.py --output-root "$PSI_RUNTIME_ROOT/SIMPLE-eval-20261103"
export SIMPLE_EVAL_ROOT="$PSI_RUNTIME_ROOT/SIMPLE-eval-20261103/snapshot-fb67cd9a"

# 另存 psi-data 的 25 份 ZIP，并解包供历史发布对照。
python3 tools/sync_simple_eval.py --release zip-38e43f22 \
  --output-root "$PSI_RUNTIME_ROOT/SIMPLE-eval-20261103"

# 显式复验，不下载、不写文件。
python3 tools/sync_simple_eval.py --verify-only \
  --output-root "$PSI_RUNTIME_ROOT/SIMPLE-eval-20261103"
```

本机 `PSI_RUNTIME_ROOT=/mnt/cfs/cqizhh/personal/huangkailun/haoxuan/data`。两个发布目录为 `SIMPLE-eval-20261103/{zip-38e43f22,snapshot-fb67cd9a}`，原始 ZIP 保存在前者 `archives/`。先同步 ZIP 时，最新快照可复用经过身份核验的相同文件。已存在但不匹配的文件会报错，保留历史原件。数据不放源码仓库，未覆盖生产训练、旧资产或评测结果。

## 实际验证与占用

- 全部 25 ZIP 完整 SHA256、成员 CRC、解包内容复验通过；最新快照 2993 文件大小与 LFS SHA256 / Git blob SHA1 复验通过。下载文件逻辑合计 123671146 bytes；网络复用不重复下载相同快照文件。
- 使用固定官方 `get_episode_lerobot()` 和实际 parquet/metadata，读取当前 17 × 4 难度 × 10 episode = 680 条，另核仓库保留的 3 份 `dr-level-hard-old` 共 30 条。共 710 条首帧记录，长度和目标/机器人空间配置保留通过。这里只用轻量 dataset facade 调用官方函数，没有构造完整 LeRobot/Isaac 环境。
- 21 项 CPU 测试通过，覆盖新下载器的错误摘要/部分下载清理、ZIP 越界路径、历史文件保留、文件复用、17 份源/镜像匹配，以及既有官方客户端、历史 RGB、HTTP 编解码。源码核验随官方集成测试通过。未进行 GPU 推理、仿真或成功率测量。
- 下载前预算 1 GB；同范围后测评测目录实际分配 253009920 bytes，审计收据目录约 0.84 MB，根缓存未变化。根盘/CFS 的 df 包含其他会话写入，不作为本次独占增量或平台镜像层验收。无依赖安装、模型/训练大包下载、清理或远端上传。

本机原始比较、API 摘要、parser 核验及前后占用收据位于用户 data 根下 `SIMPLE-release-audit-20261103/`；代码内只保留固定身份和可审阅的核验结果。
