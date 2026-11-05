# SIMPLE 官方版本替换（2026-10-30）

所属：RHSCI-Private/main 普通目录。替换 SIMPLE、安装仿真环境或比较新旧评测结果前必读。用户要求使用官方最新合并版本；本次仅本地提交，不推送远端。

当前资产入口和搜索范围更正见 [2026-11-03 更新核验](SIMPLE_UPDATES_20261103.md)。下文日期记录保留历史范围；11 月 1 日“未核验到修正”的结论只适用于所查旧包，不能用于当前资源。

## 版本与来源

<a id="zero-arm-evaluation-20261105"></a>

### 当前零臂评测（2026-11-05）

当前实验不是未经修改的官方版本：基于下文固定的 `b75c6786`，另有本地
SIMPLE 提交 `0b2d3ecedaac789a15b664d7169b5f7318b82cf5`。它在12个任务
构造函数中复制控制器配置并设置双臂零初值，两桌任务官方已设置，共13个任务。
运行时直接使用修改后的任务源码；该本地提交不在上游官方仓库，不能要求别人从上游 checkout 此 SHA。

本仓库提供完整的 [源码差异](../configs/simple/zero_arm_tasks.patch)，包含原文件上下文，
仅用于在新机器重建相同任务源码。完成官方安装、在干净的固定版本上运行
`tools/check_simple.py` 后，再执行以下一次性步骤；已经使用本地零臂提交的机器无需重复应用：

```bash
git -C "$SIMPLE_HOME" apply --check "$PSI_HOME/configs/simple/zero_arm_tasks.patch"
git -C "$SIMPLE_HOME" apply "$PSI_HOME/configs/simple/zero_arm_tasks.patch"
```

`tools/check_simple.py` 与 `configs/simple/official.json` 校验的是未经修改的官方基线，
应用差异后不能把它的失败解释为安装损坏，也不能声称此时仍是完全原版。
差异遵循 SIMPLE 原始 MIT 许可；作者及许可见外部仓库 `license.md`。

当前零臂实验的初态包为 `USC-PSI-Lab/psi-data` 固定
`e78fb93cc28912a3031a10b8656d32d7f0a2b867` 的 `simple-eval/*.zip`，
不是新版 `SIMPLE-eval/fb67cd9a`。三级各10初态、每级5轮、只录第一轮双目视频。
客户端是官方 `Psi0DecoupledWbcAgent`/`HttpActionClient`，RHSCI wrapper 调用
原生 `run_eval`；服务端是本仓库 `rhsci.inference.http_server`，HTTP `/act`/`/health`。
π0.5为30预测/30执行，GR00T为16/16，无RTC；各任务独立步限与成功阈值。
测量噪声默认关闭，开启时仅在客户端对发送前31维加噪。
具体命令及逐项配置见 [首页当前评测表](../README.md#current-evaluation-configuration-2026-11-05)
和 [推理说明](INFERENCE.md)。下文保留2026-10-30官方基线迁移记录。

- 官方：[physical-superintelligence-lab/SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE)。2026-10-30 只读核对 GitHub API 与远端 main，固定 `b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11`。
- 此提交是 2026-10-29 合并的 [PR #29](https://github.com/physical-superintelligence-lab/SIMPLE/pull/29)：两桌递物任务复制控制器配置后，将双臂初始关节角设为零，避免原抬臂初值在稳定阶段碰桌；官方通用 `g1_sonic.py` 未改。PR 作者报告的 MuJoCo 结果不是本轮实测。
- 旧 README 固定 `803db7e4c1f0fef11008c9480ee75b2f9cc0ca36`，官方比较 API 显示新版领先 28 个提交。本次采用整份新版原生源码，未将更新缩减成单个姿态补丁。
- [固定清单](../configs/simple/official.json)保存 433 个顶层仓库文件的 Git blob、SHA256、大小与 9 个官方 gitlink。SIMPLE 保持外部依赖，不复制入 RHSCI 的 `src`。官方 `license.md`（MIT）与全部组件许可证保留在原 checkout。
- 未携带历史本地 WBC、导航、渲染、水龙头或时钟补丁。RHSCI 星号模型的因果 RGB history 包装仍是显式 opt-in，不能将启用该包装的流程称为全部零修改官方链路。

## 本机实际落地

完整顶层源码和全部 9 个递归子模块位于：

`/mnt/cfs/cqizhh/personal/huangkailun/haoxuan/data/external/SIMPLE-official-b75c6786`

子模块使用 SIMPLE gitlink 指定的 SHA，未单独追踪各自 main。SSH URL 仅在下载命令中等价转为 HTTPS，没有修改 `.gitmodules`。LFS smudge 关闭；顶层 16 个图片仍为指针，子模块运行资产也未下载。本轮没有安装 SIMPLE 的 Python/Isaac/CuRobo/WBC 环境，没有另行下载训练权重或评测数据。官方第三方 Git 源码自带的 AMO jit 策略、WBC ONNX 和测试回放等文件随仓库保留，不能将源码下载记成纯文本零资产。现有 PSI0-RAW 生产源码、训练环境、数据、checkpoint 和进程不变。

README 的默认 `SIMPLE_HOME` 改为 `$PSI_RUNTIME_ROOT/external/SIMPLE-official-b75c6786`；本机令 `PSI_RUNTIME_ROOT` 指向上述 CFS data 根目录即可对应。新源码独立于旧目录；不删除或覆盖历史实验。

2026-10-30 用户要求清理旧版本：检查确认 RHSCI 当前树没有旧 SIMPLE 源码、嵌套子模块、备份或忽略的运行产物，CFS external 也只有本版 SIMPLE。已删除 `pyproject.toml` 中指向旧 `third_party/SIMPLE` 的废弃 sources/group 注释；依赖解析内容不变。当前树只保留有效接入，来源版本号作为迁移记录保留，Git 历史不改写。

## 核验与使用

在本机 `/root/workspace/RHSCI-Private`，标准 Python 3 可执行以下只读检查，数据集、模型权重、结果和训练日志均不适用：

```bash
python3 tools/check_simple.py /mnt/cfs/cqizhh/personal/huangkailun/haoxuan/data/external/SIMPLE-official-b75c6786
```

检查 HEAD、433 文件 SHA、9 个直接及所有递归子模块版本，并拒绝源码修改、暂存修改或未跟踪文件；不下载，不启动训练或仿真。输出明确标记为 source only。运行环境中的 `simple` 也必须从此 checkout 的 `src` 导入，不能只按始终为 `0.1.0` 的包版本号判定。

使用独立环境安装的供用户执行命令见 [README](../README.md#install-simple) 和[固定官方安装指南](https://github.com/physical-superintelligence-lab/SIMPLE/blob/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11/docs/source/tutorials/installation.md)。`uv sync --frozen` 保持官方 lock；本轮未执行这些安装命令，也未在未安装环境中声称真实 CLI help/完整评测可运行。下载前后须记录同范围空间，环境放代码树外的 SIMPLE 独立目录。

本轮 CPU 实际验证：

- 官方源码检查：433 文件及 9 个递归子模块通过；直接从固定 `src` 导入的路径检查通过。
- `test_simple_official`、`test_rsci_history_client`、`test_http_inference`：共 14 项通过。使用官方原生 `HttpActionClient.query_action`，通过内存 ASGI 服务验证 current RGB、32D state、reset、四帧 condition 与 16×36 动作响应的真实编解码。测试服务使用构造动作，不加载策略权重。
- 语法/格式及源码导出清单检查见本轮提交；未执行 GPU 推理、物理仿真、成功率评测、训练或完整回放。

复验上述 14 项时，设置 `PYTHONPATH=src:tests`、`PYTHONDONTWRITEBYTECODE=1`、`CUDA_VISIBLE_DEVICES=''`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`SIMPLE_HOME` 为上方源码根，使用本机 `/root/workspace/PSI0-RAW/.venv-psi/bin/python -m unittest test_simple_official test_rsci_history_client test_http_inference -v`。这是既有服务 CPU 测试环境，不是 SIMPLE 仿真环境。未设置 `SIMPLE_HOME` 时新增官方源码集成测试明确跳过。

验收曾遇到原生 GR00T-WholeBodyControl 的四个 `libddsc*.so.0` 符号链接被 `git status` 报 M，但类型、readlink 字节、SHA 与 HEAD 相同，`git diff --exit-code HEAD` 为零；具体 stat/filter 原因未定位。检查器改为分别检查 HEAD 内容、暂存内容及未跟踪文件，并逐个核对递归子模块，不忽略子模块生产源码变动，也未修改官方源码或使用 assume-unchanged 隐藏差异。

## 数据与协议边界

PR #29 另提及 level-2 episode 0/6/7/8/9 的初始手部与物体接触问题，相关物体位置修正在 eval data，**不包含在代码 PR 中**。2026-10-30 源码替换时未下载、替换或几何核验本地评测资产；2026-11-01 的独立数据核验结果见下节。

2026-10-30 官方 HF API 返回 `USC-PSI-Lab/psi-data` main 为 `e78fb93cc28912a3031a10b8656d32d7f0a2b867`（lastModified 2026-09-14），`USC-PSI-Lab/SIMPLE` 为 `0e8f23cce8f2e96717aa60a0e26bed9d56223c2d`（2026-10-28）。README 的 task zip 下载示例固定前者，资源脚本仍为官方版本；这些元数据不证明 PR 所述修正已被纳入，后续须核验实际场景/episode，不能将两个仓库混称同一资产版本。

历史训练数据、归一化、动作最后四维语义、RTC/噪声/时钟协议均不能因 HTTP 字段兼容就视为完整兼容。没有自动重写旧 checkpoint 或动作。本版评测结果使用独立输出目录并记录 SIMPLE SHA、子模块 SHA、资产 revision 和全部开关，不与旧版结果合并统计。官方客户端不自动生成 RHSCI 的 `history.psi_eval_deterministic` 元数据，直接使用时服务端保持 `--no-deterministic-eval`；只固定普通 seed 不等于验证了历史请求噪声协议。

<a id="eval-layout-audit-20261101"></a>

## 2026-11-01：level-2 位置修正的独立核验

**历史范围：当日仅核验下列旧发布包，两包均没有可识别的位置差异；本次搜索漏查了已存在的 `SIMPLE-org/data/eval-fixed/`。2026-11-03 已找到并同步新版，见[更正及新版坐标](SIMPLE_UPDATES_20261103.md)。** 本轮实际下载两份固定版本 ZIP，核对大小、完整 SHA256 和全部成员 CRC，再比较文件内容与每个 episode 的配置。不是仅按上传日期推断。

| 来源 | 固定 revision | ZIP bytes | SHA256 |
| --- | --- | ---: | --- |
| [官方 psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/blob/e78fb93cc28912a3031a10b8656d32d7f0a2b867/simple-eval/G1WholebodyLocomotionPickBetweenTablesTeleop-v0.zip) | `e78fb93cc28912a3031a10b8656d32d7f0a2b867` | 480532 | `8a4b0ff7d068967dac8baec2c4a4deca4f644215a1ee15e921739f3dfe90e7e0` |
| [RHSCI 历史交接包](https://huggingface.co/datasets/ds1x/RHSCI/blob/2bff75eb0ca574211ed95a5d88d4fc8482426369/eval-data/SIMPLE/task-layouts/G1WholebodyLocomotionPickBetweenTablesTeleop-v0.zip) | `2bff75eb0ca574211ed95a5d88d4fc8482426369` | 490093 | `ed40763fd5d2d23e121c2a2900c31c30ed70823b8ee7310bb66907adb92f2527` |

两包的 **75 个实际文件逐字节一致**；历史包另有 22 个空目录项，ZIP 打包差异不能作为位置更新证据。三个 `dr-level-*` 目录各 10 个 episode 的完整 `environment_config` 都一致，全部目标位置及朝向无变化。以下是 level-2 中 PR 指定五条的记录位置（米；两来源相同，显示舍入到六位）：

| Episode | x | y | z | 两包位置差 |
| --- | ---: | ---: | ---: | ---: |
| 0 | -0.314270 | 0.058306 | 0.763422 | 0 m |
| 6 | -0.330995 | -0.051027 | 0.763422 | 0 m |
| 7 | -0.329937 | 0.013423 | 0.763422 | 0 m |
| 8 | -0.305153 | -0.051186 | 0.763422 | 0 m |
| 9 | -0.329676 | 0.017821 | 0.763422 | 0 m |

坐标取自 `meta/episodes.jsonl` → `environment_config` → `dr_state_dict.spatial["0"]`；同一配置的 `target` 为 `graspnet1b` / `uid="0"` / `cracker_box`。直接调用固定官方 `get_episode_lerobot` 读取全部 30 条 metadata，确认它保留上述位姿；没有构造仿真环境或运行物理。原生评测把该配置交给 `env.reset(options={"state_dict": env_conf})`，代码更新不会替换 ZIP 内的目标坐标。

匿名 no-cache API 复查时，官方只有 main 分支，仍为上述 revision；两桌 ZIP 的最后提交为 `7fb4a11d4a564ecdba459586da6e0e103422b833`，时间 `2026-09-14T06:57:58Z`，早于 PR 于 `2026-10-29T05:53:51Z` 合并。结合逐文件比较，当前公开下载入口没有提供可识别的新版位置修正；未取得作者修正后的独立坐标/文件作为参考，不能反向推定移动方向或自行平移物体。

[机器可读证据](../configs/simple/eval_layout_audit_20261101.json)保留两包身份、75 个文件的 SHA256、全部 30 条目标位姿及检查范围。原始 ZIP、API 收据与完整差异报告存于本机用户 data 根下 `SIMPLE-between-eval-audit-20261101/`；只是审计副本，没有替换在用资产。本轮没有验证手与物体的实际接触、物理稳定阶段或成功率；这些结论需修正参考资产和仿真核验。无需修改 RHSCI 模型或训练数据；仅本地提交，不推送。

## 空间记录

源码下载前后记录在主树 `Psi0_Image_Storage.md` 的 RHSCI 官方 SIMPLE 两阶段条目：顶层预算 200 MB、初始化全部子模块阶段预算 2 GB；CFS 原目录保留。最终同范围占用 2,184,257,536 bytes；子模块阶段增量 2,150,416,384 bytes，超过 2 GB 的预估增量 150,416,384 bytes。GitHub 仓库 size 元数据低估了固定旧提交及其工作区/Git 对象的合计；保留原件，没有为达到预估而清理。占用含子模块 Git 自带策略/回放及 Git 对象，不是平台镜像层验收。没有制作镜像或清理共享缓存。
