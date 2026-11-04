# 错误记录

- **静态接触审计的环境/路径检查（2026-11-04）：** 辅助汇总误用系统python3，因无NumPy立即失败，未修改报告；改用既有.venv-simple/bin/python后核验通过。源码查询首次漏掉官方仓库src层级，rg报路径不存在；以报告native_builder的实际路径定位后完成核对。防复发：诊断数值处理显式使用模型/仿真环境，源码定位优先既有收据中的绝对路径。两桌physics_dt来自任务metadata=0.005，不用引擎通用默认0.002代替；本审计仅forward不积分。

- **跨材质模板筛选不能作总体新旧录像比较（2026-11-04）：** compare_video_recoils初版21px模板相关阈值0.8导致120段中87段拒绝，多数仅背景/光照变化，剩余33段有明显选择偏差，不据此判断新旧总体。首份video-recoil-distribution/result.json与日志保留。检查120格初帧后改直接腕部固定点，复用原LK前后向0.5px门限、不降低门限：左120/120通过，右112/120通过，右8拒绝及分母保留，不能当模型失败。抽出track_points前后原保存121帧三点轨迹/误差max逐位一致；Ruff初次仅import空行排序未过，修正后通过。最终结果支持新包早期换块回撤较多，但不外推完整任务/纯因果或称根因找到。

- **在线lag诊断的sensor字典检查两次拒绝（2026-11-04）：** 首版完整字典比较拒绝，二版仅补capture_depth/capture_pointcloud=False仍拒绝；逐字段核实另有intrinsics/depth_intrinsics/depth_to_color_extrinsics=None，RAW来源省略五个新增字段。改为只补这五项关闭/空值后再严格比较完整字典，width/height/fov/mount/pose/baseline等仍必须相同，不吞其他差异。online-image-joint-lag-audit.log及*-v2.log保留失败，v3及旧包分析退出0；最终带episodes SHA版本另写online-new/old-lag-with-provenance.json，不覆盖初次成功报告。防复发：区分序列化缺省字段与几何变化，不笼统放宽相机检查。此为诊断预检问题，不是生产顿挫根因。

- **小幅快照安装差异不能直接判错帧或静默当零：** FK/像素时间诊断首次预设相机/torso安装矩阵差<1e-4，实际0.00048018触发拒绝，无结果；image-joint-lag-audit.log保留。实测四份快照平移差最大0.488mm、旋转差0.00926度，改为每份独立校准分别扫描所有lag并报告敏感性，不通过放宽阈值来宣称完全刚性。最终20组最佳均lag0，1220拟合全成功/有限；结论仅限选定训练窗口，不把标定误差当数据错帧证据。

- **跨初态的开环下身目标不是有效物理基线：** 完整RAW目标诊断中，将当前WBC保存目标改从RAW初态开始，第120步躯干偏53.05度，不能拿RAW下身恢复至0.63度宣布修复。补从trace自身稳定初态重放，现场腰max误差0.003228rad；相同初态/手臂条件下换RAW腿腰目标反使第120步误差8.77→16.21度。保存目标依赖原反馈，跨状态不能当闭环控制；后续必须先验证基线，再报告完整轨迹和反例。首版contact bodyid的NumPy隐式标量转换警告用.item()修正，两次补跑无此警告，物理计算未改变。

- **原始/训练episode编号不能直接对齐：** 官方RAW episode0为844帧，与训练episode0的586帧不匹配。训练manifest实际raw_parquet指向episode78；按该路径仅读取后，其SHA与清单精确相同，586帧双臂/腰状态逐位匹配。两个已取HF版本的RAW episode0文件SHA相同，不能把编号错配误判成版本变化。后续先核完整source_binding路径、SHA、帧偏移及实值，再使用raw基座/动作；不同包装的processed_identity_offset=1不覆盖本次rerender的frame_offset=0。

- **独立USD检查版本不等于Isaac内置USD版本：** 第三次Kit已正常初始化，但其`ListEditorProxy_SdfReferenceTypePolicy`没有独立usd-core的`GetAppliedItems`，六路在场景加载前退出。输出完整移至`*-before-usd-api-compat`。计数改用两个版本均提供的explicit/added/prepended/appended/deleted/ordered列表逐项计数，不改变引用本身；第四次日志`gr00t-old-package-queue-usd-compat.log`，实际验收仍以Isaac运行结果为准。

- **pxr必须在原生Kit初始化后导入：** 第二次GR00T启动2314573中，导入SIMPLE CLI/API仍未启动Kit，修复器立即调用pxr导致六路ModuleNotFoundError，未开始物理评测。输出移动保留为`gr00t-official-pre1102-5cycles-20261104-before-kit-hook`。改为在实验入口包装原生`BaseDualSim._init_isaac`，先完整执行原初始化再修正Layer，随后才由原生框架加载场景；不在Isaac环境安装另一个usd-core、不手工抢先创建SimulationApp。第三次实际启动及验证状态见WORK_LOG。

- **GR00T三个场景USD跨目录引用及大小写错误：** 首次旧包烤箱六路预热通过，但102344280初始化出现180条props缺失警告/路，引用../../../../../../props落到运行根。旧修复器只识别/root/tmp/UID或/tmp/UID，未覆盖此种路径。已TERM队列2292579及其完整子树，将目录移动保留为*-before-oven-usd-repair，不计成绩。三个当前任务场景均有同类引用；全场景扫描75个同名不同内容，不能给运行根建共享props软链接。scene_references.py仅在当前进程内重定向至本场景原props，持有Sdf.Layer保证原生加载复用，不保存官方文件。初次CPU核查因交接black.usd不存在拒绝；仅在唯一大小写候选且文件内精确同名材质prim存在时启用black→Black及White→WHITE（对应/World/Looks/black、White），不猜别的材质。三场景79/27/202路径完成修正，引用位置总数与磁盘USD SHA保持，原生USD组合错误0。原ZIP清单/大小/CRC核对、实际Kit渲染验证分别记录，不把CPU组合成功当全部渲染通过。完整命令/收据见队列README顶部。该问题不同于此前两桌的临时目录失效别名；不得据此推翻已验两桌材质或宣称π0.5顿挫根因已找到。

每次开始工作前必须完整阅读本文件和 `AGENTS.md`。错误按实际证据记录，不用静态检查、下载成功或进程存活替代实际运行验证；修复后更新原条目中的状态与验证范围。

- **DDS不能仅靠编号不重复保证可用：** 出生点配对准备记录预留240/241，客户端尚未启动。DDS RTPS基础端口7400+250*domain，240已为67400，超过UDP65535；domain还需容纳participant端口，不能使用233以上。已在WORK_LOG提示后续选现场未占用224/225，不重启尚未开始的客户端，不将此准备错误计模型失败。GPU2梯度诊断已结束可释放给该对照；8148模型服务不使用DDS编号。
- **生产BF16梯度不能用微型CPU逐位等价代替：** 真实VGQ40k/batch1/两组合成标签，history/both对reference整体梯度差3.43%/6.11%，Reader v_proj最坏且1/5参数超已有容差；同profile重复仅约0.045%–0.518%，辅助valid=False后所有方式差落入同量级。进一步用SHA/manifest与训练契约匹配的真实ARDY标签完成14次backward，history/both差约2.48%，同profile约0.36%–0.44%，只有Reader v_proj超原容差；masked组约0.45%–0.58%且无超限。query去重算和当前/原认证both也有小型CUDA数值波动，不能据微型CPU逐位相同推及此处；所有223梯度有限、冻结/RNG核验通过。原始三份training-gradient JSON及参数列表保留，不能因此声称特定训练bug或成功率回归根因已证，不放宽容差、不改旧配方/模型。近似共享AE辅助/动作cos0.0028，不支持本样本强抵消，不代表全训练。ARDY归档首试simple环境因无zstandard在请求前退出，改用miniforge现有依赖只下载一次16.3MB、校验/原生Reader验证成功。
- **配对噪声预检等待不足，未判定服务失败：** 新独立8148服务首次原生编译超过客户端300s等待，预检ReadTimeout；现场PID2156408仍活跃且编译缓存文件持续更新。没有重启服务器，也不当模型False。原preflight.json空文件保留，verify_paired_noise.py使用独立新流verified-v2、1800s等待且不自动重试，避免旧请求晚完成后序号冲突。此前simple环境直接导入rhsci失败发生于启动前；显式PYTHONPATH指当前src后通过。首次后台PID2156261缺失/日志空，缺少终止原因证据，已改持久前台session并核对真实PID/health，不从后台返回的PID冒称服务成功。

- **梯度诊断显存预检拒绝不是模型失败：** probe_training_gradients.py首试GPU1时空闲不足16GiB，加载模型/forward之前主动抛错退出1，原training-gradient-parity.log保留。随后现场GPU1已有其他8148诊断服务、空闲8.6GiB；改用GPU2现场17.1GiB、新日志/结果路径，并保留每进程2GiB预算。不能把较早显存读数当长期资源保证、不能因此停止其他评测或宣称梯度验证已执行；实际启动和结果按两桌README更新。
- **官方SpatialDR漏恢复机器人四元数，已复现、未修复：** b75的src/simple/dr/spatial.py机器人分支只读已保存position，不读quaternion；robot_orientation_region=None时使用Actor原有朝向并回写覆盖保存状态。使用真实SpatialDR/RobotActor/Pose和实际metadata做CPU调用：旧level0十条误差0，新四难度40条全部朝向偏差0.00304073025rad（0.17422°）、位置误差0。源码SHA51b5f168…、逐条输入/恢复值/回写值保留spatial-robot-restore-audit.json。缺陷确切，但没有闭环反事实证明它是当前成功率骤降或跳变主因，不能因发现一处bug即宣布goal完成；仅前移x的成功样本也保留此缺陷。新旧160步稳定各后退6.134cm，配置额外后移6.0732cm真实保留到策略开始，记录spawn-settling-audit.json；这不证明未公开生成器“重复稳定”。脚本及实际命令在两桌README，未改生产/官方源码。

- **并行导航诊断的DDS冲突与历史标签更正：** 本轮新启动navigation-v3两路与另一并行工作刚启动的old-episode4-navigation复用了DDS220/221；随后ps确认碰撞，只终止本轮PID2063616/2063617，父进程亦已退出，目录原样保留。前者current已有稳定阶段记录、无model_chunk，不能断言未发生跨进程DDS干扰；两个目录的结果均不作为干净物理对照（不影响DDS100–107正式评测和210–212旧150次复测）。改用新目录navigation-v3-corrected、GPU2/3、DDS230/231并确认无占用后重跑。启动前的单次进程检查不能排除其他会话同时启动，后续检查实际PID、DDS和输入来源。另，尚未提交的legacy_navigation.py原实现是legacy四维范数，却标为历史v3；已读原归档环境锁及认证源码@1f96adda的rsci_between_settings.navigation_method，历史v3应为前三维速度范数<0.05。本轮校正实验适配器并记录selector/适配函数SHA；已启动旧进程不会热更新，old-episode4-navigation/legacy仍是旧四维实现，不冒充v3。生产控制器未改。

- **独立诊断DDS编号碰撞，整组结果作废：** old-episode4-navigation与另一并行navigation-v3曾同时使用220/221；后者current已进入稳定期，有串扰可能。已TERM本会话2056693/2056694、保留输出，不把该组轨迹当科学对照。正式100–107和旧包210–212未重号。启动前查进程只能证明当时空闲，不能阻止并发启动；后续单一负责会话启动导航，不重复启动。并行补查还确认历史v3选择器看前三维速度，最初四维范数只对应更早协议，不能仅从基础适配脚本推断最终历史运行语义。
- **历史manifest字段语义必须追到写入端：** 初版历史审计把stats_sha256与训练meta/stats.json比较，得到不同摘要；随后读取实际上传器rsci_now_videos.py，确认该字段是results/eval_stats.txt成绩文件摘要，不是归一化。已修正审计器为逐组成绩文件身份一致性，不报告虚假的“模型统计不匹配”；初版报告改名保留first-field-interpretation。不能仅凭字段名相似推断其来源。

- **旧包三路共享服务新增技术故障，非原卡顿根因：** GPU7原独立服务1780025第一次act为200，后续AnyIO请求线程内CUDA Graph捕获发生cuDNN设备分配失败，随后capture/mempool错误、HTTP500；三路未完成episode，不能计模型False。已停止服务及残留客户端，保留dedicated尝试完整日志。实验serve_single_thread.py复用正式CLI/协议/模型，只把所有预测提交到单工作线程；新服务1801072的预热1次＋三客户端并发6次均200、有限(30,36)，收据shared-server-singlethread-check.json。不同线程的CUDA局部上下文/内存池为待核解释，短期绕过成功不证明已查明CUDA内部原因，也不能解释原有单路视频顿挫。生产框架未改。
- **归档适配层数值诊断初始化：** 首次把当前Processor的normalization关键字传给旧构造器，旧源码只收root/tokenizer/variant，TypeError发生于推理前。已实际阅读归档构造器，确认其固定v2公式与本权重contract一致；改为显式限制v2并传旧三参数重跑，失败日志保留archived-adapter-comparison-first-init.log。不能靠同名类推断旧API签名。

- 旧包完整复测的依赖更正：短探针复用原模型服务可行，但五轮完整复测可能超过原worker剩余生命周期，模型会被原worker正常清理。首次准备进程已停止、日志保留、不计失败；改为GPU7/8147独立同权重服务。不能把单条旧初态120步诊断当作复现或否定旧包约30%历史成功率的证据。官方旧ZIP与RHSCI镜像ZIP虽摘要不同，解包75文件全同，须区分容器身份与实际内容。

## USD 引用旧路径失效：已修复，全部 40 初态真实渲染与桌面绑定通过

- 现象：实际 Isaac 渲染日志出现 `Could not open asset @/root/tmp/bdb1f316aa0188ec/props/TV_Screen2.usd@`，同时还有 Shader11、MIRROR 等 USD 子资源引用错误。不能把它归为不影响场景的普通启动噪声。
- 原因：官方场景 USD 内含 `/root/tmp/<编号>/props` 等旧绝对路径；本机该路径仍是指向已删除旧运行目录的失效软链接。官方加载器准备的 `/tmp/<编号>/props` 及当前官方源资源实际存在，但不会自动修正 `/root/tmp` 下的旧链接。
- 遗漏：此前修复只扫描 `/tmp`，没有覆盖 `/root/tmp`；默认日志级别还会省略部分警告。材质文件存在、烟雾测试能出图，不代表 USD 组合完整。
- 处理：`experiments/pi05_between/repair_usd_paths.py` 扫描官方 HSSD USD 中实际引用的 `/tmp`、`/root/tmp` 路径；仅将失效链接改名保留，再建立指向当前官方源资源的兼容链接，不改官方 USD 文件。正式评测客户端开启 warning 级别保留引用警告。
- 验证：88 处别名均存在；四难度各 episode 0 完成真实 reset/render，逐 prim composition errors 全为 0、图像均为有效 `(360, 640, 3)`。`logs/pi05-between-material-rendering-textures-restored.log` 中 `Could not open asset`、缺失纹理、`UsdToMdl`、`Failed to resolve`、traceback 全为 0；`logs/pi05-between-material-rendering/result.json` 保存结果，level-2 图像已人工查看。范围是四个初态的实际渲染与全部 40 初态的静态 MDL 检查，不宣称所有后续 rollout 已完成；后续仍检查正式日志。
- 防止复发：扫描资产中真实嵌入的所有绝对路径，而不是只猜一个临时根；同时检查 MDL 文件、导出名称、纹理、USD 子引用与实际图像。保留首次失败日志及旧链接，明确验证覆盖哪些初态。
- 补充发现：USD crate 的引用可能被压缩，`strings` 返回零条不能证明没有引用；实际相对路径可解析至 `/root/tmp`。修复脚本已改用 `Sdf.Layer.GetExternalReferences()`，解析并恢复了 44 处场景引用别名。渲染诊断使用了 Isaac Sim 4.5 未提供的 `Stage.GetCompositionErrors()`，又因 Kit 的 `close()` 掩盖 traceback，导致初次看不到失败原因。现改为逐 prim 读取 `GetPrimIndex().localErrors`，先保存结果或 traceback，再关闭；不要拿其他 USD 版本的 API 当作本环境事实。
- 为避免错误场景计入评测，已主动停止首次 8 卡启动，并将完整输出保留为 `evals/pi05-between-30x30-5cycles-20261104-before-usd-repair`。这次尝试不计入正式 400 次。
- 分层复验继续发现 `../textures/*.png` 缺失：只恢复 props 会使 USD composition errors 清零，却仍缺背景纹理。修复已改为同时恢复每组 props 与 textures；必须再看 `[UsdToMdl]` 和资产解析日志，不能只用 composition errors 判定材质正常。
- 后续完整复验：`logs/pi05-between-material-rendering-all40/result.json` 含四难度各 10 个初态，40 次真实 reset/render，USD composition errors=0；80 处 table/table2 有效绑定检查均与官方 layout 的 MDL 路径/导出名一致，无材质的 table2 未额外绑定。`logs/material-all40.log` 的缺失 USD/纹理、UsdToMdl、traceback 均为 0，程序已正常关闭；level-1 episode 7 图像已人工检查。官方 ground_material 字段未实际应用，保持官方实现，没有自作主张增加地面随机材质。

## 下载节点与流量：专用节点暂不可用，已按用户授权回香港续传

- 用户要求以后大文件走下载专用节点。发现日本 Y05/Y06 均标注“下载专用 | x0.01”，此前 HF/CDN 的实际连接链却是香港 Y01。已增加独立下载组和优先域名规则，保留普通流量原选择，未把节点标签当作实际套餐计费验证。
- Y05 TCP 超时，Y06 DNS 解析失败；两个节点的 HF 实际 HEAD、HF robots、gstatic 测试全部失败。已暂停准备/下载进程 267176、267262、267263，并仅断开旧 HF 连接，不自动回退普通节点。文件和分片保留；专用线路恢复且实际小请求验证后才能续传，不能将规则配置成功写成下载已恢复。
- 最新复查与用户授权覆盖上段暂停条件：Y05 DNS 成功、TCP 仍超时；Y06 本机解析失败，但 Google/Cloudflare 两个公共 DoH 都能解析，绕过本机 DNS 直连其 A 地址端口仍超时。说明并非 HF/权重故障，也不能把 Y06 简单断言为域名失效；服务端或中间线路具体原因尚未证实。香港 Y01 TCP/HF 测试通过。用户明确“能下载才是重要的”后，已加入香港为下载组首选、原生配置校验/热加载/选中成功；真实烤箱 STATE 权重 64 KiB Range 返回正确 206、长度与 Content-Range，随后核对身份并恢复三个原进程，保留续传，未恢复无关下载。
- 新规则初次与原 YAML 的单空格列表缩进不一致，原生校验失败；两次失败均未加载。改为与既有列表一致后 `mihomo -t` 通过，才用本地 API 热加载。后续修改先保留受限权限备份、确认列表缩进、原生校验，再加载；不打印含认证信息的代理配置。

## RTX 5090 首次 π0.5 编译候选内核不支持：真实推理已通过

- 现象：Torch 2.7 / Triton 3.3 的 max-autotune 预热中，部分候选报 `computeCapability not supported`、`PassManager::run failed`，日志显示 cc=120。
- 已知范围：原生模型严格加载通过；编译调用仍在执行，栈检查确认在图处理/代码生成中，并非仅凭日志推断卡死。候选失败不等于整个推理失败，也不能在实际 `/act` 返回前称兼容验证通过。
- 处理：保留原推理公式与编译设置，完成真实请求验证；没有通过删日志、换采样步数或假模型掩盖错误。
- 后续证据：8 个模型均实际返回有限的 `(30, 36)`，原生首次编译预热通过，无需切换编译 backend。候选内核失败与完整推理失败必须分开判断。

## 其他仍需保留的已知限制

- **短窗口改善不能外推为完整修复：** 新包episode4前移0.06m在180步窗口边界比14.03→6.63，并在完整749步取得True；但成功轨迹全程该比值11.65，与原位失败11.85接近。episode0同样前移后虽搬运到第二桌仍最终掉落失败。距离变化是影响因素，不能被包装成唯一顿挫根因或所有episode均修好。历史成功视频选定二维窗口也有回撤，不能仅比较最终成功标签判断动作连续性。
- **历史执行长度的实际反例：** 已追溯同一权重历史服务确为预测30执行24、无RTC，但旧level0 episode4真实180步对照30/30与30/24的边界/块内跳变比均约12.3。两路完整预测及前缀/动作映射已核对，不能仅发现历史参数不同就宣称顿挫根因是执行30步，也不能擅自更改正式30/30。原生随机输入采样不构成固定噪声配对，结论仅为这两条轨迹均未消除回跳。
- **根因与资源对照范围更正：** 上轮将动作块回跳的直接机制当作可结束 goal 的结果，且将用户“旧评测数据”误解为只检查原场景模型/材质。6014 个资源文件相同不能证明旧 task-layouts 的初态、物体/机器人位姿、相机和布局配置相同。已按用户更正继续 goal，独立解包旧 HF task-layouts，真实核对已发现位置/布局变化，并进行旧数据闭环对照；不能用文件哈希核对替代这项实验。
- **反事实不能只报告支持项：** 真正换旧初态后 VGQ-AE 仍回跳；灯光恢复到训练值 50000 后也仍回跳，不能凭肉眼明暗差宣布灯光根因。训练初态下减弱但未消失。稳定阶段后退约 6.1cm 已被官方 PR29 写明是零臂初态后的 WBC settle；两组前移出生点的诊断在策略前把盒子碰落，不构成恢复成功。全程仿真时钟也只是把跳变延后一帧：边界本帧小，下一帧完整尖峰，120步最大WBC跳变0.3045rad/实际单关节单步0.0277rad。没有据此修改正式配置。原机器人/目标物7008文件与HF清单SHA相同。所有反例输入/动作/视频完整保留，不能只看改善的一帧。
- **故障层级与结论限制：** 同链专家动作回放的边界/内部目标变化比为0.97，关节重现误差很小，图像基本匹配原记录；同一真实评测图像中，STATE只换状态即可把首目标由当前姿态退回初态，VGQ-AE视觉预测始终偏早期。证据支持当前权重的视觉闭环反馈不足，不证明某行训练代码错误或全部框架无bug；不同模型权重不是方法消融，不能给VGQ-AE临时加未训练的状态token。初步对比不同DR图像时怀疑相机/亮度，后续同初态对照没有支持相机错误，必须保留后续反例并更正判断。
- **直接用训练数据作初态诊断的额外开销：** 官方 get_episode_lerobot 会读取整条轨迹的所有字段；此数据含 360×640 深度嵌套列表，首次生成约 26GB HF Arrow 缓存，随后逐帧 hf_transform_to_torch 很慢。py-spy 栈已确认，不是 DDS/推理卡死，最终正常完成。后续仅初态诊断可用官方轻量评测记录承载逐项相同的训练 environment_config，必须核对配置相同并声明不是专家动作回放。26GB 缓存保留，未删除训练数据或既有结果。
- **诊断工具路径：** activate.sh simple 会 cd 到官方 SIMPLE，后续仓库脚本必须用绝对路径或显式 cd 回本仓库；一次相对路径分析命令因此未执行，绝对路径重跑通过。Ruff 使用仓库要求的 0.17.2 缓存可执行文件，simple 环境自带 0.15.7 会被 required-version 拒绝，不能把拒绝当检查通过。

- **动作一卡一卡：直接机制已复现，尚未宣布修复。** 对用户视频反馈不能仅做 FFmpeg 解码、帧数或进程检查。三个同初态 120 步实际诊断显示第 30/60/90 步新块的模型双臂目标回跳；两次 VGQ-AE 的边界中位 L2 跳变为块内中位的 21.25/23.04 倍，STATE 对照约 4.12 倍。原始模型动作到 WBC 双臂输入逐项最大误差为 0，真实 MuJoCo 关节随后反向；这已足以解释本样本的周期性顿挫，而非仅有视频容器错误。推理约 0.068–0.090 s，每步仿真固定推进 0.02 s；没有证据把录制视频中的反向归因于等待推理。状态/物理角差小于 1.2e-6 rad，输入图像随姿态变化。证据在运行根 `evals/stutter-diagnostic-20261104`，配置与命令在必读两桌 README。
- **诊断时钟的反例：** CPU 重放原生插值器，改成仿真时钟会使第 30 步尖峰看似变小，但下一步出现更大的完整跳变；必须查看边界前后整个窗口和真实运动，不能只挑边界一帧宣布修复。本轮未改控制时钟、模型动作、执行长度或资产，也未用平滑隐藏问题。MuJoCo qpos 为树遍历顺序，左右臂分别为 22:29/36:43，不能套 motor 的 body29 顺序使用 22:36；分析器已按官方模型核实，避免把手指错当右臂。图像更新和本样本映射一致不等于所有兼容问题均已排除；模型为什么重规划回到较早姿态仍需区分学习行为与其他输入/推理原因。原版场景资源替换按用户要求留到最后，未执行。
- **独立数值诊断初始化：** 首次额外对照采用 meta 构造后 assign 权重，但 rotary 非持久 buffer 不在 checkpoint，`.to(cuda)` 报 `Cannot copy out of meta tensor`；尚未开始对照推理，正式服务未受影响。实验脚本改为与服务相同的普通构造再严格加载，避免遗漏非持久 buffer；不能将此诊断脚本错误归为在线模型加载失败。首次日志保留为 `inference-mode-comparison-first-init.log`。
- **额外对照已完成：** 同一 VGQ-AE 权重、四张真实输入、相同 seed=17 噪声，eager / max-autotune 的双臂最大单关节差 0.004512 rad，最大联合 L2 差 0.009599 rad，存在 BF16 执行差异，不声称逐位等价。eager 在第 30 步也从实际左肩 -0.11909 rad 预测到 -0.03072 rad，仍会反向；本样本的大回跳不能以关闭编译解决。原始场景 `102344022` 与全部材质共 6014 文件，和 RHSCI 原资源 manifest 逐文件 SHA256 及文件名集合均一致，缺少/多出/不同为 0，未做无实际版本差别的资源替换。具体权重学习/其他兼容因素仍不能仅由一条截短轨迹全部排除；记录已证实的直接机制，不编造已修复的程序 bug。

- 本次静态审查及 CPU 复现发现等待队列的故障检测缺口：`experiments/simple_queue/run.py::current_complete()` 在检查 worker PID 前，遇到缺失的 `COMPLETE.json` 直接返回 False。如果当前 worker 意外死亡，队列会继续等待而非显式报错。当前 8 个 worker 均存活，尚未观察到此故障发生；已记录，未在仅要求检查的任务中改动故障处理流程。不能把一直等待当作评测仍健康运行，后续检查须同时查 PID 与官方统计进度。

- 单卡两套独立 π0.5 + Isaac 容量探针：GPU 2 上第二个 STATE 真实预热 `(30,36)` 通过，但场景加载峰值占用达到 30336 MiB，余量 1774 MiB；触发预设 2 GiB 保护，只结束额外 worker。未完成诊断 episode，不能称“双路正式评测通过”，也不能把保护触发说成已经 OOM。原 8 路继续运行，暂不扩大双路。额外 Isaac 初始化约 300 秒，原生 Kit 文件明确记录等待 `RtPso async group async compilation`；不能只凭控制台停在 app ready 就诊断死锁或 USD 错误。原始过程在 `evals/dual-gpu2-capacity-probe-20261104/`。
- 材质清单复核不能按目录外观擅自扁平化：官方 `vMaterials_2.zip` 自身同时包含顶层及嵌套 `vMaterials_2/` 成员，内外 `material_split.npy` 内容也不同。新核验脚本第一次擅自归一化成员路径，误报 2757 项多出及 split 冲突；已改为严格按 ZIP 原始成员路径比较。复核 50 个官方包、34068 个文件，缺少/多出/内容大小或 CRC 不符均为 0，88 处别名目录也均完全一致。没有因此移动或删除任何官方资源。前述“嵌套目录修复”指恢复加载器需要的顶层路径，不表示官方所有嵌套资源都应删除。
- 8 个 Isaac 默认各开数百线程，本机 112 CPU 的 load 曾超过 600；多路在 `_wait_for_viewport` 等待，不能只看 app ready 宣称开始 rollout。现按 NVIDIA 官方运行参数限制每实例 Carbonite/TBB 为 8，首轮 8 个客户端均已完成启动（部分约 190–200 秒），已有实际 30 步动作执行。此前尝试保留为 `-before-thread-limit`。线程限制是资源配置，不改控制频率/预测步数；未把所有等待都归因为已证明的缓存错误，辅助隔离缓存探针已停止，没有据未完成探针修改官方环境。
- Isaac `close()` 可能直接结束 Python，不能依赖关闭之后的结果写入，也不能只看退出码 0。实验编排已在每轮客户端退出后，核对官方 `stats/eval_stats.txt` 中 episode 0–9 恰好各一条，再写 `verified_results.json`；不足 10 条即失败，不写完成标记。补充核验前的一次预热输出保留在 `evals/pi05-between-30x30-5cycles-20261104-warmup-before-result-check`，尚未进入实际评测。
- vMaterials_2 曾被恢复脚本额外套一层目录，已调整至官方 `resolve_data_path` 期望的 data 根；归档和解压收据保留。仍需与上述 USD 子引用问题分开检查。
- 官方篮子 USD 的碰撞 mesh 没有 points，Isaac 会报不能建立该碰撞；本路径的物理由 MuJoCo 执行，但不能据此声称 USD 资产无错误。此问题不是材质缺失，未猜改官方资产。
- 官方 WBC/MuJoCo 手指映射历史问题、四项成功判定公开代码缺口仍按 `SIMPLE_UPDATES_20261103.md` 披露；两桌不属于那四个成功判定缺口任务。
- π0.5 保留历史 v2 分位归一化语义及其已知信息损失，不替换旧权重统计。推理无需 optimizer、RNG 或训练源码完全相同；不能为普通启动增加这些门槛。
