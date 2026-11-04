# 必要工作记录

- 作者咨询材料补实际推理代码：报告新增HTTP→processor→Eagle/DiT四步Euler→反归一化→16×36响应→官方WBC调用链、源码路径、NextGr00t历史标记/当前AutoModel注册、STATE与VGQ去state-token分支及修改边界。仓库外logs/gr00t-oven-author-inference-20261104.zip已生成，174文件/434866字节，ZIP CRC及内置逐文件SHA复核通过，归档SHA1e83dddbf9e6be622239b450fd8069bda0a1cf6d407429e3b3ef6b5437dab123；含保存配置/实际命令和当前及启动时scene脚本，非完整安装包、未发送外部消息。

- 用户补充旧评测包/场景边界后复核：旧烤箱ZIP75文件仅json/jsonl/parquet/mp4，不含USD/纹理/MDL二进制；旧environment_config控制场景选择、灯光、材质引用及初态，资源独立来自USC-PSI-Lab/SIMPLE@0e8f23cc，102344280 ZIP SHA88b98529、336个非目录文件与原ZIP集合/大小/CRC一致。尚未证实与作者历史旧评测资产逐文件相同，不把旧配置包称全部资源回滚。作者报告已补独立资源版本与明确待问边界。

- 已按用户请求整理docs/GR00T_OVEN_AUTHOR_REPORT_20261104.md并登记AGENTS：09:48:49 UTC快照STATE成功1/0/0/NA、完成40/41/43/NA（未完成），VGQ-AE成功34/23/23/NA、完成50/50/50/NA（已完成）；包含版本/归档SHA、保存训练配方、32D状态/36D动作、原生95%eval裁剪、16/16/4步BF16无RTC、WBC墙钟、成功判定累计逻辑、材质修复范围、实际命令、视频和作者问题。明确论文表8多任务不可直接作单任务复现对照，初始化身份未确证。只整理材料，未发送外部消息或更改/重启评测。

每次开始工作前必须完整阅读本文件、`AGENTS.md` 及 `ERROR_LOG.md`。这里维护后续工作所需的当前事实；实验具体参数和完整命令维护在对应已登记文档，运行日志和大产物放在仓库外。所有状态区分准备、启动、实际 rollout 和完成。

## 当前仓库与环境

- 工作仓库 `/root/workspace/RHSCI-Private`，分支 `main`；仅本地提交，不自动推送。模型使用现有 `src/<模型>/rhsci` 和公共 `src/rhsci`，实验编排放 `experiments/<主题>/`。
- 运行目录 `/root/workspace/Haoxuan/RHSCI-runtime`，`/opt/rsci-data` 指向此处。`source <运行目录>/activate.sh pi05` 选择模型环境，`source <运行目录>/activate.sh simple` 选择仿真环境；激活脚本会改变工作目录。
- 8 张 RTX 5090，每卡约 32 GiB。SIMPLE 是官方 `b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11`，9 个递归子模块固定在 gitlink；官方跟踪源码未修改。Isaac Sim 4.5.0.0、MuJoCo 3.3.6。59 个官方场景/机器人/材质归档已恢复。
- 模型环境 `.venv-pi05-rsci`：Python 3.10.12、Torch 2.7.0+cu128、Transformers 4.53.2，已安装仓库 overlay。8 个模型均通过原生严格权重加载及真实 `(30, 36)` 推理预热，无需更换编译 backend。

## 当前 π0.5 两桌任务

- 本会话恢复后第3次核对同一授权缺口，工具update_goal实际返回blocked：用户尚未明确放行4条旧包π闭环诊断，AGENTS第32行停止要求仍有效；纠正后的DDS180–183/PYTHONPATH命令已备好，拟输出目录仍不存在，未运行。既有接触/限位、初距分布、120视频总体对照及新旧在线lag五份报告均在，尚无完整根因证据，不标完成；继续配对闭环需要用户明确允许该诊断范围。该状态覆盖上方/下方历史active快照，正式GR00T评测不因根因目标blocked而停止。

- 待授权历史组合诊断执行前检查完成：修正尚未运行方案的DDS240–243为180–183（基础端口52400–53150），补simple激活后PYTHONPATH仓库src；CPU实际验证无该路径find_spec(rhsci)=None，补后action_noise协议导入成功。提取真实CLI在首次写配置/仿真导入之前的parser及校验，四条旧level0ep4/level2ep8参数、episode存在、端口范围均通过；输出目录仍不存在，未启动模型/DDS/物理/π评测。训练/采样源码复读未新增根因证据；旧包当前30/30对历史30/24+sim+legacy的4条闭环已在README备完整命令，需用户明确放行，不能越过既有停π要求。准备错误不是历史顿挫原因，当前目标未完成；本轮具体进展是纠正待执行方案，授权缺口按本次恢复后首次记录，不直接blocked。

- 原生静态接触/限位核对完成（session50730退出0，CPU、无模型/物理积分/评测）：旧/新episode4稳定末态＋前120步首次手臂环境接触为视频101/112帧，早期30/60/90回撤均在此前；旧level2 episode8全121帧无手臂环境或自碰撞，仍三次目标cos约-0.999。四条另含新前移6cm均无采样自碰撞，14臂关节实际/目标距限位最小0.815/0.722rad（旧停滞1.127/1.041rad），不支持碰撞或机械关节限位为这些回撤的必要原因。只50Hz姿态mj_forward，不能排除中间瞬时接触/力矩饱和；不是完整根因/修复。audit_recorded_contacts.py、recorded-contact-limits-boundaries.json/log保存源/资产/输入SHA与逐帧结果，Ruff及四组121帧/输入SHA/前版数值复核通过；完整实际CPU命令、范围在两桌README顶部。本会话工具get_goal实际active，继续只读排查，不恢复π评测、不动GR00T。

- 已CPU核对91训练配置与新旧三级初态：训练configured target−robot x=0.30450–0.33537m，新30/30超出（0.36856–0.39305m），旧level0/1全在范围、level2仅2/10超出；不可称全训练帧支持范围或稳定后实测。RAW78训练0首帧真实距0.39552m、旧/新episode4稳定末态0.37515/0.43604m，四校准目标原点投影y差约20px，未新增仿真/模型。各包level0/1机器人/目标完整pose20/20对相同但场景/干扰物/外观改变、回撤不同；各组距离/回撤描述相关无一致单调，距离不能单独判完整根因。audit_spawn_conditioning.py及spawn-conditioning-audit-v2.json保存SHA/逐初态统计/四校准，完整命令及范围见两桌README新节；无生产修改，goal active。

- 120段VGQ40k新旧30/30既有录像完成CPU腕部换块分布核对：各包三级前两轮60段，左腕全部通过，前121帧各级60边界窗≥1px反转旧19/31/38、新35/51/54，≥2px旧12/21/15、新22/46/51；两组各180块内对照窗≥1px均0，支持用户新包更易回撤观察，不外推整段/纯场景因果。右腕8段拒绝显式保留，右减左后各级趋势仍在；全部视频/日志/配置SHA与原始轨迹见bilateral-wrist-recoil-with-provenance/result.json，session78420退出0。compare_video_recoils.py与复用track_points、完整命令/表在两桌README新节，Ruff通过；初版模板87/120拒绝不可作总体判断已记ERROR_LOG。无新模型/仿真/生产修改，根因未定。

- 新旧包已有在线轨迹各121帧完成CPU 3D图像/真实关节lag核对，无新π模型/仿真。两组四校准×五刚体最佳均lag0；同帧指尖/手腕留出RMSE新0.2611/0.1837px、旧0.2410/0.2441px，±30帧新2.42–3.60/旧5.10–9.10px。官方稳定后writer初帧规则和0/30/60/90模型PNG同编号视频最小MAE均核实；仅支持这两窗口无整块在线滞后，不能反驳旧包整体顺畅观察/外推总体或称完整根因。audit_image_joint_lag.py新增可选在线输入，两个online-*-lag-with-provenance.json与输入SHA/命令在两桌README新节；sensor检查两次拒绝及有限缺省等价记录ERROR_LOG，未改生产配置。

- 本轮CPU只读优化器与冻结骨干核对已完成（sessions47026/53236/70245退出0）：NO-STATE/STATE各208个可训练参数张量、430106660元素，保存model源码SHA7946e111匹配各契约；原训练源码与当前meta构造的参数名/顺序/形状/冻结标记完全一致。实际optimizer.pt均208槽、全部step40000、动量有限/二阶非负、无整张量零动量，最终lr1e-8；不证明中间恢复无同形参数置换或学习有效。三模型603个冻结VLM张量共2923335408元素逐位相同。VGQ远端optimizer按Range尝试因正确model repo返回401未读取，不能冒称223槽已验；受限pickle解析器在本地NO-STATE真实文件与torch.load的208槽/步数逐项一致（58470有效字节），远端范围缺口保留。三新脚本Ruff通过，无GPU/训练/模型更新；optimizer-payload-{nostate,state}.json、frozen-backbone-identity.json和远端失败日志保存，命令/范围在两桌README顶部，具体根因未确证、goal active。

- 新CPU真实FK/训练像素lag分析已退出0：RAW78完整实录基座/关节＋现有视频手部跟踪，固定30–89帧、5帧块交替拟合/留出、lag±30；四份SDK安装校准×五候选刚体共1220拟合全部成功/有限，20组最佳均lag0（不是20独立轨迹）。index指尖/wrist_yaw同帧留出RMSE0.289/0.217px，±30帧约5.75–17.13px，不支持本窗口整块错帧。快照安装差0.488mm/0.00926度显式作四份校准敏感性分析；初版刚性阈值拒绝日志保留，不称完全刚性。audit_image_joint_lag.py及image-joint-lag-audit.json、完整命令/范围在两桌README顶部，Ruff通过，无新模型/仿真/正式配置修改。全数据/完整根因仍未定。

- rate/flag原始数据审计和episode8导航门限单变量物理重放均完成：RAW78第三维与heading闭环角速度相关0.9990，前120帧target_yaw恒0但第三维75帧非0；全91轨迹21845帧非零但abs<0.1。官方f169793将返回rate改flag、5b304e4再加flag门限。episode8真实trace原生输出逐项复现，门限影响44/120次、转速差max0.04813rad/s、站走选择不变。原生CPU MJ/WBC重放相同VLA目标，基线43关节max误差3.17e-5rad；仅取消门限后双臂max差0.000298rad、基座0.112mm/0.1674度、躯干0.2183度，短窗口物理影响有限，不能判大幅手臂回退根因。session79485/88559退出0、DDS226/GPU6上下文释放；前两次环境/预检失败详ERROR_LOG。rate-flag-semantics-dataset-and-episode8.json、episode8-gate-physics-v2.json保留；命令/范围在README顶部。无生产修复/新正式π评测，goal仍active。

- 已只读比较旧包episode4/30-30与新固定噪声原位/前移6cm三条已有轨迹，前三边界旧跳变0.1537/0.1337/0.2551rad、新原位0.1649/0.1642/0.2124、前移0.0901/0.1003/0.1379。前91步均未持续抬高盒子8cm；前180步旧/新原位双臂净移动0.678/0.696rad相近，盒子升高0.171/0.00578m，不能把新失败等同手臂没有净推进。持续抬起首步旧159/前移103，新原位1200步无。compare_approach_progress.py及approach-progress-comparison.json保留配置/trace SHA、逐边界保留投影及范围，命令见两桌README新节、Ruff通过。旧/新噪声未配对，旧180步诊断False非完整失败，个例不外推旧包整体顺畅程度。无新模型/评测/生产修改，完整根因仍未定。

- 新增纯CPU原生PD完整RAW目标重放及两种初态对照，三个进程均退出0/DDS230释放，无新π0.5模型调用。RAW初态+RAW完整目标腰RMSE0.000189/0.000637/0.000227rad、第90步躯干0.518度；但不能和换初态后失稳的WBC保存目标53.05度比较来称修复。恢复trace自身初态后基线腰相对现场max0.003228rad，腿腰15维换RAW目标使第120步躯干误差8.770→16.206度、base x RMSE9.97→61.02mm，反证直接移植原始下身目标不是修法。两份*-paired-replay/trace-initial-replay.json及日志保留，脚本/完整命令在两桌README新节，Ruff通过。当前下身目标对初态/反馈敏感，视角差仍不能单独判控制bug；新旧包差异与完整根因未定，goal仍active。

- 真实训练forward/eval动作分支6组对照已退出0（GPU7/PID2465327/session24084结束）：VGQ40k、训练episode0帧30/60/90、seed17/29、原生历史图/增强/Beta时间；捕获实际训练动作embedding及当前图，train重复、eval同cache、eval重新编码三路全部逐位相同、最大差0。仅no_grad离线前向、ARDY目标无效占位，无训练/服务/仿真。action-train-eval.json/log保留，脚本audit_action_train_eval.py Ruff通过，完整命令及范围见两桌README顶部。不支持模式切换制造回跳，未证明全训练正确或找到根因，goal仍active。

- 用户否定腰部/相机追因方向后，本会话转查无状态模型共享训练链路。新CPU原始轨迹核对已退出0：固定HF1885a61a的RAW78与训练0，586帧完整32Dstate/36Daction最大差9.54e-9/1.15e-7，未见本条标签转换错位；RAW/action与state列序不同，按各自名字映射。新增Range只取该条parquet和视频。两种渲染器121帧手部跟踪前后向误差<0.183px，固定30–89窗口水平位置最佳lag0，但速度最佳-28、指标不一致，不能称绝对同步验收。91条元数据只68条唯一匹配，其余不定性。raw-training-alignment-episode0-motion-v2.json与完整命令/限制在两桌README顶部；两份新脚本Ruff通过。无新模型/仿真/正式评测，无生产修改，goal仍active、具体根因未确证。

- 已找到训练episode0对应的官方RAW episode78（非0），仅HTTP Range新增738904字节parquet，HF固定79baf58、文件SHA c41a9683…与训练manifest完全一致；586帧双臂/腰states逐位相同，绑定offset0。新增CPU audit_recorded_base_camera.py用完整实录基座/关节核对原生FK：专家动作回放第90步躯干角差5.07666度，只修腰仍4.97216度、只修基座3.01548度；此前RGB-D补偿剩0.65663度/躯干位置21.22mm，支持角度估计但非完整位姿真值。完整命令/收据recorded-base-camera-factorial-audit.json在两桌README新节，Ruff通过，无新模型调用/评测。用户指出旧包基本顺畅，不能用个别旧样本回跳外推新旧相同；新旧总体差异及完整根因仍待解释，goal保持active。

- 相机+95%裁剪的残留误差已复用analyze_sampling_bias.py作逐种子分解（只读已有预测，无新模型调用）。第60步新首误差0.00589–0.01630rad、旧末0.03301–0.06308rad，新首有符号投影仅0.028–0.175；此处不能继续统称新首严重回退，第30步两端均偏。渲染图三种子边界误差均值平方占总平方89.16%，相同报告原训练图36.44%，支持剩余共同输入偏差，非总体统计。新增可选--center-crop-fraction=.95，默认不裁剪结果与旧boundary-origins-vgq.json完全一致；实际两份新收据corrected-crop-origin-bias.json/default-crop-selection-verified.json保留。必要命令/限制在两桌README增强节，Ruff通过。未修改生产推理/控制；根因仍未完整确证。

- 同一CPU分析器已进一步只读复核独立会话完成的static-camera-render两份预测：固定95%裁剪，相机位姿恢复后第90步跳变0.15179–0.15738→0.01081–0.01946rad，新首误差0.16081–0.16887→0.00903–0.01441rad，cos约-0.99→-0.246/0.249/0.041；第30/60步仍有0.04395–0.10683rad跳变。支持相机视角为第90步的重要条件因素，非完整根因/闭环验证；相机为RGB-D估计且包含专家腰角。两个*-boundary-audit.json保存原始文件SHA与逐边界数据，未新增模型调用。GR00T烤箱VGQ旧level0首个episode已官方True，双目录像闭合；人工查看0/1.5/3/4.5秒拼图确认拉开烤箱过程，logs/gr00t-old-oven-first-success-review.jpg。该单例不外推总体成绩，当前日志无缺失USD/UsdToMdl。

- 只读已有72份裁剪预测的逐边界CPU审计已完成，未启动新π0.5推理/评测：训练原图95%裁剪后三种子九边界绝对跳变均降低，块内中位变化很小，改善真实；但渲染图第90步三种子绝对跳变均增大为0.14423/0.15535/0.15938rad，新起点误差0.15391–0.16567，远大于旧终点0.01252–0.02586rad，回退cos约-0.98至-0.997。第60步裁剪后也均转负方向，不能以三边界中位数称完整修复。新增实验分析器audit_crop_boundaries.py、center-crop-boundary-audit.json保存身份/SHA/逐边界误差分解；命令在两桌README原生增强节。Ruff通过，根因仍未确证，goal保持active，正式GR00T旧包队列不变。

- **当前GR00T队列2329010/session37378活跃，日志`logs/gr00t-old-package-queue-usd-compat.log`。** 六份warmup均有限(16,36)，六路Kit内修复各79处原场景引用，已开始原生episode初始化/双目录像；现场日志缺失USD/纹理/UsdToMdl/Traceback均0，尚无完成episode。不据初始化通过宣称全部渲染/完整评测验收。第三次2321277因Isaac内置USD缺少GetAppliedItems退出，输出完整保留为`*-before-usd-api-compat`；改为两个USD版本均支持的原始list-op计数，CPU三场景复验仍组合错误0（logs/gr00t-old-scene-reference-audit-listop/result.json）。完整错误已记ERROR_LOG。下文旧PID均为历史快照。

- **当前状态覆盖下文所有旧PID快照：** π0.5/NO-STATE均已停止。GR00T首试2292579因场景引用错误停止，输出完整移至`gr00t-official-pre1102-5cycles-20261104-before-oven-usd-repair`；二试2314573因Kit初始化前调用pxr失败，输出完整移至`*-before-kit-hook`，均不计成绩。现修复为包装原生`BaseDualSim._init_isaac`、先调用原方法再修正进程内Layer，保留torch/curobo/Kit原生顺序。第三次队列2321277/session97966、六worker2321278–2321283已启动，日志`logs/gr00t-old-package-queue-kit-hook.log`；当前仅确认模型加载，实际渲染与完成结果待核。原场景三个ZIP共1240文件的集合/大小/CRC32全部匹配，收据`logs/gr00t-old-scene-original-inventory.json`。命令及完整配置见队列README顶部。

- **最新用户指令覆盖以下历史运行状态：停止所有π0.5评测，不测新包，仅用官方旧包评测此前指定且下载完整的GR00T四组（烤箱STATE/VGQ-AE、交接STATE、水龙头STATE）。** 已TERM队列420186，原NO-STATE四worker及模型/仿真子进程均正常清理退出；旧π0.5父1812071/2/3已TERM，客户端2195095/2195572/2223472已INT，8147/8148服务1801072/2156408已TERM，现场ps无上述评测/服务残留。已有文件保留，不自动恢复，不把中断算失败，logs/evaluation-scope-change-20261104.json记录用户中止身份。新调度仍复用旧worker，只增加显式evaluation-root；plan只含GR00T及三级，每组150次，共600次，hard=NA。三旧任务ZIP仅1.177MB已下载并HF SHA校验、三级各10初态核验，权重不补下载。**新队列2292579已实际启动，工具session52445，日志gr00t-old-package-queue.log；六worker2292580–2292585、服务2292967–2292972均活跃，六份真实warmup.json确认有限(16,36)，首轮仿真正在初始化，尚无完成成绩。** 首波烤箱STATE GPU0–2 / VGQ GPU3–5，8120–8125、DDS100–105；后续交接STATE再水龙头STATE。完整配置（内部50、有效16/执行16、4次去噪BF16无RTC）、来源和命令见队列README顶部。

- 用户指定旧包level2/repeat01/episode8停滞已补同初态120步诊断（session71467退出0，GPU7/DDS227释放，既有8148服务保留）：4块模型0.068–0.093s，双臂边界回跳为块内中位26.864倍，与上一块推进方向cos约-0.999，真实关节跟随回撤；模型→WBC臂映射差0。120步盒子最大移动0.904mm、基地0.792mm、下身全部stand。原两轮录像都见类似停滞，首轮日志40×30动作/1201帧，非推理停止。直接机制是反复起手/退回，不将其说成完整训练根因或材质因果。analyze_stall.py及完整来源/命令/范围在两桌README，运行根old-level2-episode8保留trace/JSON/图/双目视频；无生产改动。

- 最新现场复核：原400次已全部完成，8份COMPLETE.json齐；STATE成功0/0/0/0、VGQ-AE成功0/1/2/0，均完成50/50/50/50。前两轮每模型/难度/轮均20份非空、带success/failed后缀的闭合双目MP4，共320份（本次核文件数量/大小，不冒称再次全量解码）。队列420186实际活跃，已启动NO-STATE GPU0–3四难度；不要重复启动。旧150次父进程1812071/2/3仍活跃，现场成功12/5/2/NA、完成24/23/21/NA，非最终。
- 独立会话固定噪声出生点episode4两路现均已出官方统计：original False、plus006 True。仅只读复用analyze_control_trace.analyze解析完整trace，原位1200步40块比12.1971、前移804步27块比12.6705，动作到WBC双臂差均0；收据paired-spawn-noise-independent-summary.json位于运行根。不操作其8148服务或目录源文件；不同完成时长/阶段的全程比不能当作出生点对顿挫效应量。成功仍伴随显著边界跳变，支持将任务出生位置影响与顿挫区分，非总体成功率结论。

- 本会话静态渲染8组已退出0（session64250/PID2252182），DDS226释放；16张双目PNG/result.json完整，实际SDK相机第90步差3.0155度/6.35mm，腰观测误差小于5.1e-7rad。拼图已检查，无新材质缺失；静态原图与此前动态回放像素MAE4.90–6.90，必须比较同次静态原姿态与修正腰。两组96次离线推理也已退出0（session39877）、GPU0诊断释放，原生VGQ40k/30预测/10步BF16/noRTC；重复训练输入全部逐位一致。静态原姿态不裁比10.61/11.57/10.76，修腰后8.01/12.43/8.48；原姿态95%比10.81/8.14/6.93，修腰加裁剪后5.47/5.69/2.82，RMSE仍0.03095–0.03170rad，未恢复训练图裁剪1.19/1.65/1.71及0.00565–0.01221rad。腰视角有影响但非完整解释，原图第90步修腰甚至预测首肩更偏；intervention-summary.json与3个新诊断脚本保留，完整来源/命令/范围见两桌README。无生产改动，完整基座/图像剩余差异待定位，goal仍active。

- 已按用户追问实际核官方配套Psi0当前main=cda3fdfa固定Git对象：PyTorch增强固定95%裁剪位置随机，训练forward train=True、sample_actions train=False。所以训练裁/推理不裁也属于官方默认，不能单凭此称RHSCI漏步骤或修复bug；中心裁剪改善仅为本权重输入尺度敏感证据。固定官方源码/README链接与当前行号已录两桌README，未更改部署配置。网页缓存行号与当前API对象有差异，源码以固定提交为准。
- 用户追问增强：已实际核保存training_augmentation及forward/Processor/native preprocessing，训练裁剪比例恒95%，只随机起点；224→212→224约放大5.66%，并非[95%,100%]随机尺度。所谓“原训练图”是盘上原图，未经过训练forward增强；正常推理不裁。尺度干预改善支持输入敏感，不证明增强本身错误或完整根因。必要语义已补两桌README，正式配置未改。
- 固定噪声出生点对照现已实际启动：服务2156408/8148继续，客户端PID2189396原位GPU2/DDS224、2189397前移GPU3/DDS225，工具session4868，输出paired-spawn-noise/{original,plus006}。此前预检session50974退出0，preflight-verified.json确认两独立流各index0/1四次真实请求200、同输入完整动作逐位相同；服务日志两对噪声SHA相同。240/241仅早期错误预留，从未启动（端口越界），已改合法224/225并现场查空闲。客户端正在初始化，不能称已完成rollout/成功率。最新正式STATE完成50/50/50/50成功0/0/0/0，VGQ完成50/48/49/48成功0/1/2/0；旧三路父进程仍活跃。禁止在上述资源重复启动。实际完整命令已在对话，README更新为224/225。

- 原生增强48次离线对照已退出0（session88316）：训练RGB原图边界比9.3924，五组native增强降为1.4001/3.5351/4.9727/1.8140/4.2722；专家RMSE从0.01894降至0.00745–0.01587rad。渲染RGB仍9.08–15.24、RMSE0.03205–0.03769，不能称完整修复。原报告native-augmentation-paired.json SHA03ec98dc…与summary保留。新增线索是训练/原图处理差异，但未分离几何/颜色，不据单episode单noise确认根因。
- 中心裁剪72次离线预测已退出0（session80578）、GPU0释放：同VGQ40k/四时刻/两图源、noise17/29/43/10步BF16。原训练RGB边界比9.392/7.505/7.733，97.5%中心裁剪后2.248/1.438/1.501，95%后1.194/1.646/1.712；RMSE从0.01894–0.02700降至0.00565–0.01221rad。当前渲染RGB95%后仍6.723/14.977/8.515、RMSE0.03124–0.03412rad。确认这些输入的尺度变化有实质影响，非旋转/颜色或horizon改变造成；不把常规训练增强本身宣布成bug或正式修复。原报告center-crop-paired.json/summary保留。用户问预测30是否过高：已解释30/24仍回跳且离线原图有偏差，预测/执行不同，正式30/30未改。
- 边界起点/终点分解已补充analyze_sampling_bias.py并实际生成boundary-origins-vgq.json：渲染RGB第90步三个seed的新块首目标双臂误差0.1432–0.1463rad，而旧块末目标误差仅0.0108–0.0175rad；此边界主要新起点预测错误，非旧块30步过冲。seed17左肩旧末-0.17366对专家-0.17185，新首-0.13500对专家-0.16995rad。原训练RGB同边界反而旧末误差较大，不能泛化所有边界；报告精确验证jump误差=新首误差−旧末误差，带符号投影不是因果百分比。配置/命令/限制见两桌README，具体渲染输入偏差来源仍未确定，goal未完成。
- 视觉前缀缓存诊断已完成两次真实进程（首轮session86977、补充10014均退出0），GPU0释放，无DDS。audit_prefix_cache.py同VGQ40k/原训练4帧/seed17/BF16：8组t=1/.1联合eager与分离eager速度场相对差0.28%–0.52%，SDPA/eager差0.35%–0.83%，KV逐张量调用前后不变。补充完整10步两入口采样，原生联合仍有边界/块内比9.8293（当前9.3924），两路径双臂RMSE0.00048864rad、最大差0.00225258rad，支持本样本不由KV分离/缓存实现制造跳变；不以局部对照证明全模型正确。首版源码e2e5ae6及prefix-cache-native-joint.json、补充sampling.json/summary保留，完整来源/配置/命令见两桌README。没有生产修改，训练偏差成因仍未确证、goal未完成。
- 用户所指旧场景已明确为11月2日合并前、仅三级的官方旧评测包，非训练数据或单独USD替换；历史VGQ-AE约30%作为参照。旧包150次独立复测仍活跃，现场阶段成功9/4/2/NA、完成17/17/15/NA；采用用户30/30，与已认证历史30/24不同，不能仅凭成功率差归因场景。新版正式STATE成功0/0/0/0完成48/48/49/49，VGQ成功0/1/2/0完成49/45/47/45，均未最终完成。
- NO-STATE40k离线24次推理已退出0、GPU0释放，无DDS/新增服务。原训练图接续/块内变化比9.497/7.579/8.293，专家回放渲染图16.086/14.498/15.563，与VGQ接近；不能把顿挫独归ARDY/历史合批。实际CPU safetensors另验NO-STATE811键、VGQ826键（多15个辅助键），8个动作/时间投影张量均不同，排除本次两个路径误载同一权重；非全参数审计、非严格训练消融。paired-render-training-nostate.json、nostate/vgq-paired-chunk-audit.json与nostate-vgq-weight-identity.json保留，来源/完整启动及分析命令见两桌README；后续200次正式队列仍等待当前400次。具体训练成因未确证，goal未完成。
- 并行出生点诊断注意：GPU2的本会话梯度进程已全部结束，可按原计划使用GPU2/3；预留DDS240/241不能使用（7400+250*240=67400已超UDP65535，合法domain最大232且还需留participant端口）。两路客户端尚未启动，后续应改用现场未占用的224/225；勿沿用下方最初240/241预留。服务8148不受此项影响。
- 正式ARDY生产尺寸梯度对照已完成14次实际backward、退出0、GPU2释放：新补16.3MB标签归档SHA619b1d31…/93成员，manifest d209f886…与训练契约相同，原生Reader/data context及91episode标签验证通过；仅当前两桌数据、无其他权重下载/optimizer/参数更新。真实frame60/四历史帧/batch1/BF16/同seed，history/both对reference整体梯度差约2.48%，仅Reader v_proj超原容差；同profile重复约0.36%–0.44%，关闭辅助后各profile约0.45%–0.58%且无超限。223个梯度均有限、VLM冻结、RNG一致，当前/认证原both差在重复波动量级，动作loss逐方式相同。近似分解共享AE加权辅助/动作范数比0.298、cos0.0028，未见本样本强抵消；差分含CUDA波动，不能当全训练因果。training-gradient-formal-ardy.json/summary及之前两份合成报告、0c4113d/0e90346源码和全命令保留两桌README；脚本Ruff通过，具体根因未确证，goal仍active。最新旧包成功6/2/2/NA完成14/14/13/NA；正式STATE成功0/0/0/0完成45/46/47/47、VGQ成功0/1/2/0完成47/43/44/43，均非最终。
- 梯度诊断前两阶段也已完成：GPU2 PID2161457/2165463分别10/21次backward，合成标签history/both梯度差3.43%/6.11%，同profile0.0449%–0.5178%、masked后回到波动量级；原始JSON/log及0c4113d/0e90346源码保留，真实标签结论以上方为准。首个GPU1预检拒绝日志保留、未加载模型，不计模型失败。GPU2不再预留；各阶段配置/完整命令及限制见两桌README。
- 本会话固定动作噪声出生点对照服务PID2156408实际活跃，GPU1/8148、工具session72407，首次真实请求仍在原生编译；请勿在GPU1重复加载，已少于16GiB空闲。预留GPU2/3仿真、DDS240/241（尚未启动），目录paired-spawn-noise。新版episode4原位与仅x+0.06，同40k/30-30/noRTC/10步BF16/wall/1200/.4，seed17逐块配对，完整双目录像/trace。paired_noise.py按stream复用原生RequestActionNoise与单线程服务，不改生产源码；客户端要求单episode trace。simple无rhsci安装，启动显式PYTHONPATH指当前src；CPU客户端绑定/编号检查及Ruff已通过。首次后台PID2156261缺失、空日志，改用前台持久session后确认真实模型加载；第一次预检客户端300s超时但服务仍生成编译文件，未重启服务、不计失败。verify_paired_noise.py以新流verified-v2和1800s超时执行同输入四请求验证，工具session50974，待真实输出后才启动客户端；保留原空preflight.json及日志。

- 旧包首轮三级30次现已完整，compare_matched_results.py实际逐编号对齐：旧包成功4/1/1/NA，新包首轮0/1/1/NA、历史首轮4/3/0/NA，各三级完成10/10/10/NA。净计数下降集中level0，但旧与历史成功身份不同、level1仍偏低；不能把全部回归归为单一平移。原生噪声未配对、历史30/24且协议不同，仅描述首轮，不当150次最终结果。matched-first-round-results.json保留输入SHA/全部30行/四格计数，完整命令及限制见两桌README，Ruff通过。三路旧包父进程1812071/2/3现场仍活跃、已进入下一轮，不重复启动。

- 离线flow条件诊断PID2149625已退出0、GPU1释放：同VGQ40k/原训练episode0四帧/三seed/10步eager BF16，共12组，加hook与不加hook输出全部逐位相同。第90帧t=1当前图干净动作双臂估计RMSE均值0.02271rad，固定同一加噪输入只改初始图prefix为0.10957rad，证实模型利用视觉；t=0.1当前图专家路径0.00381rad，自主路径0.02301rad，偏差仍未纠回。不能把低t缩放误差当训练loss复现或代码bug证明。flow-conditioning.json/summary/PNG保留、图已检查；probe_flow_conditioning.py Ruff通过、完整启动命令/来源/范围在两桌README，无生产修改/训练/额外权重下载。另实际导入五个patched Transformers模块，实际文件SHA与契约全同，五个声明依赖版本全同，收据installed-transformers-contract-audit.json。旧包首轮三级已各10次，成功4/1/1/NA、完成10/10/10/NA；正式STATE成功0/0/0/0完成42/42/43/44，VGQ成功0/1/2/0完成44/40/41/40；不是最终结果，具体根因未确证，goal仍active。
- 新增只读审计完成：audit_spawn_settling.py从旧/新episode4真实160步稳定记录测得各后退6.134cm，新配置另后移6.0732cm，策略前目标前向距离37.515→43.604cm；只证明位移叠加，未发现公开生成器代码，不能称重复稳定bug。audit_spatial_robot_restore.py直接调用原生SpatialDR/RobotActor/Pose，复现机器人只恢复position、漏读并覆盖保存quaternion；旧level0十条无误差，新四难度40条均0.00304073025rad（0.17422°）朝向误差、位置误差0。已记ERROR_LOG，未改官方源码或做修复反事实，尚不能解释主要回归。两个实际JSON收据、完整命令/来源/范围在两桌README；脚本Ruff通过，goal仍active。

- 相机动态时序诊断PID2138037已完成：旧level0/episode4/120步/同VGQ8147，GPU0/DDS222已释放，输出camera-timing-vgq。完整160稳定＋120策略、4块；双目各280次观测的callback/direct帧差和world减reference时间均0，相邻图像SHA无重复，4次模型请求SHA精确对应紧邻上一观测。不能把Isaac步进0.08s与MJ0.02s不同直接当图像延迟，也不以元数据证明逐像素/骨骼绝对同步。边界回跳比仍13.3971、模型到WBC臂映射误差0；图已人工检查，双目各121帧50Hz闭合。实验analyze_camera_timing.py实际运行、Ruff通过，命令/来源/范围在两桌README；不计120步False为正式模型失败。最新旧包150次阶段成功4/1/1/NA、完成9/9/9/NA；正式STATE成功0/0/0/0完成41/41/42/43、VGQ成功0/1/2/0完成43/39/40/39。仍未确证具体训练/部署根因，goal继续active。
- 认证历史训练记录已核对：只补读同一VGQ已选权重约1.3MB trainer_state与resume_origin元数据，SHA/HF Git blob均匹配manifest，未补下载任何其他权重或optimizer。记录40k/91.3242epoch，5类loss/grad/LR各4000条网格完整且有限；后10k action_loss中位0.004833、ARDY0.06211。原train.py SHA83d1c66c…匹配契约，4000条LR全部精确匹配原公式(log_step-1)，最大误差0，未见日志中的调度重置。7000步迁execution=both保留payload只是原收据声明，未加载optimizer/RNG，不扩称数据顺序正确；低带专家加噪输入的训练loss不证明自主采样/闭环连续性。audit_training_history.py、training-history-lr-audit.json及完整命令已实际完成并写入既有两桌README；无生产更改。
- 手指兼容只读审计已完成：认证bfffc460/eval_wbc_hand_order.py SHA f30f8dbe…，真实MJCF/WBC RobotModel＋原生from_psi0/PD，对plus006-episode0的1200个原始动作和同物理位置/零速度，历史“旧动作兼容＋具名输出”与当前全部关节PD力矩最大差0Nm。反馈槽位确有最大0.80006rad错位，但真实_build_wbc_observation→compute_observation的1200组86维下身网络输入差0（不读手指）；上身插值不读反馈。未推进物理/插值，不冒称闭环成功率反事实，不能仅凭手指补丁未迁入宣布回归根因。audit_hand_contract.py及hand-feedback-contract-episode0-verified.json已实际运行、Ruff通过，完整命令/范围在两桌README。旧包三路仍活跃，当前成功4/1/1/NA、完成8/8/8/NA；与新版首轮共同episode0–7成功4/1/1对0/1/1，非配对噪声且不是150次最终结果。
- 完整任务阶段已按真实物体位置量化：spawn-full原位episode4从未抬升（最高z0.76832、初始0.76275m），机器人仍空手移动；前移6cm的episode4第105步抬升、第497步近篮子、官方True；前移episode0第146步抬升、第544步近篮子，却第1046步落地、官方False。不能把两类失败均称抓取失败，也不能把原位盒子z0.68872m误写掉到地面。只读实验analyze_task_stages.py，连续5步几何阈值、trace SHA/官方统计原文及图保存在task-stage-audit.json/png，图已检查、Ruff通过。未启动仿真或改生产参数。原150次旧包及正式400次继续，goal仍active，具体训练/部署代码根因未确证。
- 固定种子误差分解已完成：仅复用paired-render-training的24次真实预测，逐边界验证平均平方误差=均值平方+样本方差。原训练图/回放渲染图边界误差中三样本共同均值偏差占88.40%/95.77%，均值动作仍有0.15649/0.08488/0.05384和0.22407/0.03981/0.14053rad跳变。支持这批样本主要是稳定条件预测偏差，不是仅换随机种子即可消除；不把三种子当总体期望或置信区间，不执行平均/平滑。实验analyze_sampling_bias.py、sampling-bias-audit.json及完整复算命令已实际验证，Ruff通过，文档沿用两桌README。
- 完成v2训练标签往返实测：91轨迹/58637帧动作、同Processor.normalize和正式服务反变换，未改任何公式/统计。三个诊断episode双臂标签RMSE0.00000440/0.00870747/0.00054308rad；VGQ对实际截断标签的RMSE仍0.01533/0.03704/0.01039rad，21个专家自然边界经往返基本相同或减小。v2确有信息损失（episode1明显），但它没有直接制造这些块边界尖峰；不能据此宣称对学习无影响或擅改v3。脚本audit_training_labels.py及收据training-label-roundtrip-audit.json已实际完成，Ruff检查通过，完整命令/来源/范围在既有两桌README。
- 正确历史v3导航诊断已完成：navigation-v3-corrected/current与historical-v3，两路新版level0/episode4、120步/4块、同VGQ40k/30-30/noRTC/10步BF16/wall，GPU2/3、DDS230/231，PID2076437/2076436均已退出。恢复历史前三维速度选择器后仍回跳，边界比当前17.9583、历史9.4920；两组随机输入非配对，不能归因比例改善或算成功率。真实lower_input同状态调用原生/历史get_action，现场协议的速度/反馈/网络选择逐项复现（每组160稳定+120策略）；稳定期完全相同，策略期站走差0/2次、转速最大差0.04913/0.06266rad/s，证实导航语义差异但不是充分解释全部回跳。trace含选择器及适配函数SHA；完整数值、实际命令和范围在两桌README，各两份闭合MP4及control_analysis图保留。最初DDS冲突两目录均作废，详见ERROR_LOG；未改生产模型/控制器，正式和旧150次评测继续。旧150次最新阶段成功4/1/1/NA、完成7/6/6/NA；正式STATE成功0/0/0/0、完成38/38/39/40，VGQ成功0/1/2/0、完成40/36/37/36，仍非最终结果。

- 已完成原训练输入三轨迹接续审计：复用已有24块/21边界，不新增推理或仿真。VGQ-AE逐episode边界/块内变化比8.7596/7.6463/5.0911，STATE为2.5987/2.1505/1.6860；VGQ整块专家RMSE0.01533/0.03861/0.01042rad。每边界专家自然变化、预测接续和当前真实状态偏差保留recorded-chunk-boundary-audit.json，源数据SHA可追溯。不是闭环，不是方法受控消融、未固定噪声，不能外推所有91轨迹。新实验分析脚本analyze_recorded_chunks.py只读已有预测、Ruff通过，配置/命令/范围见既有两桌README；未新增文档。
- **覆盖下两条导航运行状态：** 本会话old-episode4-navigation两路2056693/2056694与并行navigation-v3诊断短暂复用了DDS220/221，已有并行current进入稳定期，不能排除串扰。已向本会话两PID发送TERM、保留全部日志，整组作废、不用其成功标签或跳变比下结论。正式100–107与旧包210–212未重号。另一个会话正在补查历史v3选择器（前三维速度范数），本会话最初适配是旧v2四维范数，不能冒称manifest对应的完整历史协议；保留源头差异，不覆盖并行修改，不再启动重复导航进程。
- 导航两路已实际启动：PID2056693/2056694、GPU0/1、DDS220/221、输出old-episode4-navigation/{official,legacy}，不再仅为准备。独立CPU安装确认原方法摘要及三处AST匹配通过；无trace时CLI以退出码2拒绝诊断开关、未导入Kit。最新旧包阶段成功4/1/1/NA、完成6/6/6/NA，正式STATE成功0/0/0/0、完成37/38/38/39，VGQ成功0/1/2/0、完成39/36/37/35；均非最终成绩。
- 导航单因素诊断由当前工作执行，不重复启动：旧level0/episode4、180步、30/30无RTC、10步BF16/wall、同8147，GPU0/1、DDS220/221，输出old-episode4-navigation/{official,legacy}。原项目bfffc460的导航适配源码与当前原生方法SHA152a1919…吻合；仅实验trace可开启三处旧导航语义，正式入口默认不变。需要真实轨迹及stand/walk网络调用验证，不以参数不同直接宣布根因。命令在两桌README和对话。
- 历史成功录像的66/72/78帧也有选定二维手部回撤：背景平移补偿后指尖/腕部方向余弦-0.911/-0.711；121帧双向跟踪最大误差0.1758px，带跟踪点抽帧已人工检查。收据historical-videos/historical-ep4-motion.json。只限该二维窗口，不推导三维关节或全部旧录像；track_historical_video.py及完整复算命令已写入两桌README。出生点完整时限成功episode4的全程跳变比11.6531，原位失败11.8459，不能把前180步改善外推成顿挫已消除。
- 全训练集时间结构审计完成：91轨迹/58637帧，视频帧数、r/avg帧率50Hz、parquet episode/frame索引、动作/状态形状与有限性、timestamp=frame/50全部通过，最大时间戳误差0。不能据此证明采集源RGB/物理状态绝对同步，但没有发现固定帧率/索引错误。`training-timing-phase-audit.json`还记录配对预测的最近专家姿态：90帧渲染图三seed首动作最接近专家59/56/56帧，原训练图为85/85/82帧；块内部基本沿专家轨迹推进，各块起点的进度偏差不同，不符合统一固定帧移。脚本audit_training_alignment.py复用实际parquet/ffprobe、Ruff通过，不改模型/数据。旧150次复测仍运行，最新阶段成功4/1/1/NA、完成6/5/5/NA。另有并行工作修改实验eval_decoupled_wbc.py/trace_control.py并新增legacy_navigation.py、track_historical_video.py，本轮保留、不纳入本轮提交、不启动重复导航试验。

- 固定噪声图像配对已完成：`paired-render-training.json`共24次真实推理，VGQ-AE40k、同一专家episode0的0/30/60/90、seed17/29/43、10步eager/BF16。90帧训练图左肩首目标约-0.175rad，当前渲染图约-0.132rad，真实回放-0.186rad，渲染视图会放大早期姿态偏差；60帧反向，不能认定全局渲染错误。更关键：离线直接在原训练RGB上预测再拼接，无仿真控制参与，边界/块内变化比仍9.39/7.50/7.73（渲染图15.09/12.73/14.38），专家自然边界0.00938/0.01278/0.00393rad。当前权重本身在该训练轨迹已缺乏分块连续性，具体训练成因尚未查清。完整值、命令、范围在两桌README；脚本新增配对选项、Ruff通过，GPU1已释放，未改生产代码或在线服务。
- 最新完整时限出生点诊断三路均已退出：新版episode4原位False、仅x+0.06为True，episode0前移仍False。保留反例，不外推普遍恢复。旧包全量复测仍运行，阶段VGQ成功4/1/1/NA、完成5/4/4/NA（仅13次，非最终成功率）；原400次阶段STATE成功0/0/0/0、完成35/36/37/38，VGQ成功0/1/2/0、完成38/34/35/34。goal保持active，尚未确认全部根因。

- 出生点完整时限对照已有官方判定：`spawn-full/original-episode4/stats/eval_stats.txt` 为 `episode_4: False`；`spawn-full/plus006-episode4/stats/eval_stats.txt` 为 `episode_4: True`，对应双目视频已带success后缀。这支持新版仅恢复前向6cm可救回一个完整任务；原生噪声不同且仅一对，不能外推所有难度或唯一根因，更不能宣称已消除动作块跳变。episode0前移对照在该快照仍未结束。仅诊断，正式400次配置未改，goal未以此标记全部根因查清。

- 独立完整模型补充对照已完成：GPU1依次构造归档STATE适配器和当前适配器，各自严格加载同一VGQ-AE40k真实权重，四张真实闭环图像、seed17、10步eager/BF16，处理器张量及全部30×36输出逐位相同、最大差0，结果 `evals/stutter-root-cause-20261104/archived-adapter-parity.json`。仍共享当前安装原生骨干，STATE归档不是VGQ训练快照；认证VGQ证据以另条为准。脚本 `compare_archived_inference.py` Ruff lint/format通过，进程正常退出，未改变正式服务。另用原训练RGB在专家回放同刻0/30/60/90补查询，结果 `recorded-replay-aligned-inputs-retry/predictions.jsonl`；训练图/当前渲染图VGQ误差并非同向增减，单次随机采样不足以认定渲染根因，配置与完整命令在两桌README。

- **认证历史记录的新证据：** 已只读访问原GitHub，Pi05-QSAFE-VGQ-AE视频manifest的60条（前两轮）全部权重SHA51f037…与当前一致；成功level0/1/2=`7/7/0`，完成=`20/20/20`，14/60；不外推全部五轮。历史实际服务预测30但返回actions[:24]，无RTC，另外启用sim-step时钟、具名手指旧动作兼容、legacy导航。manifest stats_sha256为eval_stats.txt成绩摘要，非模型归一化统计。完整Git blob校验收据 `evals/stutter-root-cause-20261104/historical-pi05-manifests.json`；原仓库仅只读，未修改/推送。
- **认证原VGQ适配层数值对照已完成：** 原分支feture/ARDY_PI05_QSAFE的HEAD1369f3e9…，外部稀疏检出external/Psi0-pi05-QSAFE-reference。模型/处理器SHA分别5a6cc3…/93150b…均匹配VGQ训练contract。4张真实闭环图像、seed17、双eager10步BF16，两版Processor全部输出相同、30×36动作最大差0。原骨干gemma_pytorch也与contract摘要一致；对照共享当前原生骨干，不扩称历史环境全链路等价。结果authenticated-adapter-comparison.json。STATE快照对照也0差，保留单独收据，不再冒充唯一来源。
- **执行长度诊断已完成：** 旧level0 episode4（历史成功）两个独立180步探针，预测30执行30/24、同一VGQ8147、无RTC、官方wall时钟，GPU0/1、DDS213/214。两者边界/块内跳变比12.3423/12.2444，仍明显回跳；不能以24步作为消除顿挫的修复。逐块验证完整预测30步、执行精确前缀及实际边界30/60/…或24/48/…，双臂动作到WBC误差均0、观测对MuJoCo误差均小于6e-7rad。输出 `evals/stutter-root-cause-20261104/old-episode4-horizon/execute-{30,24}/control_analysis.{json,png}`，双目录像保留。两次原生随机采样/渲染不是固定噪声配对，不外推成功率；未给150次旧包或400次新包改为24。
- **新包出生点单因素180步诊断已完成：** 新旧level0 episode4目标位置相同、机器人出生x相差0.060732m。新版仅x+0.06的fixture十条配置除x外递归一致；原位/前移边界跳变比14.0284/6.6259，仍回跳，但前移组120步已提起盒子，180步z=0.90724m，原位组z=0.76281m仍在桌面，视频已人工对照。两组策略前目标均稳定在桌面，未碰落。不能把此前训练初态+0.06/新版+0.12碰落的失败当作此特定变化的反例。产物 `evals/stutter-root-cause-20261104/new-episode4-spawn`。训练91条初态目标前向距离范围0.30450–0.33537m，旧level0为0.30783–0.33182m，新为0.36856–0.39256m（均出生配置，非稳定后实测）。单次原生随机对照不证明唯一根因或完整任务成功。
- **出生点完整时限验证已启动：** 新episode4原位/前移、episode0前移，均同8147/40k/30-30/无RTC/10步BF16/1200步/0.4，GPU0/2/3，DDS217/218/219，PID1935502/1935500/1935499，输出 `evals/stutter-root-cause-20261104/spawn-full`。三路保存视频/trace，完整命令已在两桌README及对话。不更改正式初态、不计入原400次。新版自带pre-fk备份的robot配置在level0/1全部等于旧包，level2有1条不同；上游00bab43e（10月30日）明确标题为十二个Teleop任务迁移出生点，11月2日才同步到psi-data。距离变化与来源记录在spawn-metadata-audit.json，不把有意迁移直接宣称为代码bug或重复稳定已经证实。

- **最新旧包对照状态覆盖下方早期启动记录：** 原dedicated共享线程池服务在首次act后CUDA捕获/500，已停止PID1780025及残留仿真，失败目录全部保留、不计模型失败。改为实验单推理线程服务PID1801072，GPU7/8147，7次真实请求含三客户端并发均有限(30,36)/HTTP200，随后实际评测持续200。三个新仿真PID1812091/1812092/1812093，GPU4/5/6、DDS210–212，输出 `evals/pi05-vgq-official-pre1102-singlethread-20261104`，阶段成功2/1/1/NA、完成2/2/2/NA（只有6次，不报最终成功率）；首个旧level0成功视频已抽帧人工检查。完成150次后需关闭这个额外服务。配置仍40k/30-30/无RTC/10步BF16/1200步/0.4、每等级5轮、前两轮双目录像。命令及失败详情见两桌README/ERROR_LOG。原八路不变。
- 此前归档STATE适配层对照已完成，后又补充上方认证VGQ源码对照，均输出0差；不可再沿用“结果尚待JSON”的早期快照。历史metadata锁的全局RTC设置被Pi05实际入口覆盖为无RTC、执行24；最终以实际入口/receipt为准。另一个45/150为OFFICIAL-Pi05-RGB且后续标记状态协议错配，不替代用户所述VGQ约30%的历史证据。

- 旧包完整对照最终使用独立VGQ-AE服务GPU7/8147（PID1780025已health就绪），三个仿真GPU4/5/6、DDS210–212（PID1784701/1784700/1784699已启动），产物 `evals/pi05-vgq-official-pre1102-dedicated-20261104`。首次共享服务启动已技术中止并保留日志，避免原worker提前关闭模型；不混入成功率，不影响原八路。初态加载中，尚无完成episode；初始日志无材质引用失败，已知篮子collision mesh无points保留。独立模型服务须在旧包三等级五轮全部结束后关闭；切勿关闭其他模型服务。全量旧包身份对照收据 `evals/stutter-root-cause-20261104/official-old-package-comparison.json`。下方GPU1积分对照“进行中”为此前快照，现已完成，10/20/50步都未消除回退。

- 用户最新明确旧场景指官方11月2日同步前、仅三个等级的完整评测包，且VGQ-AE历史约30%（历史权重/命令未核实）。已下载官方e78fb93旧包SHA256=8a4b0ff7…90e7e0，解包75文件与此前镜像旧包全部相同；不能以单条120步探针代替完整旧包成功率。已准备独立完整复测脚本run_old_eval.sh：三等级各10初态×5轮=150次，30/30无RTC、10步BF16、1200步/0.4，前两轮双目录像，复用VGQ8124–8126，GPU4–6/DDS210–212；命令与来源见两桌README。固定噪声10/20/50步积分对照已完成，step30首目标仍约-0.03而实际-0.119，不是已验证修复。

- 用户明确要求继续找根因，并更正“旧评测数据”指旧 task-layouts 的初态/布局等；上轮模型输出回跳只是直接机制，不能当根因已查清。持续 goal 已重建。旧初态已实际完成 VGQ-AE 120 步闭环，仍回跳（边界/内部比 13.08）；灯光恢复训练值也未消除（20.74）。训练完整初态为8.79；前移出生点会在策略前碰落盒子，官方PR29已记载约-6.1cm为WBC正常settle。全程仿真时钟也只延后尖峰。专家动作回放已完成且换块无额外尖峰（比值0.97），30/60/90步真实双臂相对训练记录RMSE为0.00083/0.00276/0.00624rad。同图交换状态证明STATE可随当前关节保持进度，VGQ-AE当前图像却仍退向早期姿态；故障已收窄到本权重的视觉闭环反馈不足，尚未确认具体训练成因或可修改的部署代码bug。GPU1固定噪声10/20/50步采样数值对照进行中，在线服务未动。配置、来源与完整命令见必读两桌README，产物 `evals/stutter-root-cause-20261104`。正式八路模型/初态/材质/时钟均未改，临时专家服务8140已关闭。

- 用户报告动作卡顿，已启动 goal 并完成三个独立 120 步真实诊断：同一 level-2/episode-3，原 VGQ-AE、STATE 输入对照、VGQ-AE 输入对照；只复用现有服务，DDS 196/197/198，均已结束，未中断正式八路。配置仍预测/执行 30/30、无 RTC、10 步采样、50 Hz；实际完整命令与来源沿用必读实验说明。产物根 `evals/stutter-diagnostic-20261104`，含逐步 trace、输入图、视频、数值 JSON 和轨迹 PNG；截短诊断不计正式成功率。
- 卡顿直接机制已定位到模型动作块边界目标回跳：两次 VGQ-AE 边界跳变约块内变化的 21.25/23.04 倍，STATE 对照约 4.12 倍；原始模型动作到 WBC 双臂输入逐步最大误差均为 0。输入观测与 MuJoCo 双臂角最大差小于 1.2e-6 rad，图像随姿态更新；实际机械臂跟随回跳发生反向。不能仅凭这一初态前 120 步判定所有场景、权重和训练均无问题；具体学习/兼容原因未确证，不宣称已经修复。
- 本轮只加实验性 opt-in 记录与分析，不改正式模型、场景、控制时钟或 30/30；也不加 RTC/平滑。用户指定原版场景资源对照为找不到其他原因时的最后手段，尚未执行。原生插值器重放显示改用仿真时钟只是尖峰延后一帧，不是有效修复；详见错误记录。
- GPU 1 独立编译数值对照已完成并退出（原 PID 508834，正式服务未改）：固定 seed=17 对同一真实输入比较 eager / max-autotune，四张图的双臂最大单关节差 0.004512 rad、最大联合 L2 差 0.009599 rad；并非逐位相同，但不足以解释 0.21–0.29 rad 的边界回跳。第 30 步实际左肩 -0.11909 rad，eager 下一块起点 -0.03072 rad，也会明显反向。完整命令见两桌 README，结果 `evals/stutter-diagnostic-20261104/inference-mode-comparison.json`；首次 meta 初始化错误及失败日志保留，后续实际对照正常结束，不把候选 cc=120 报错单独作为整次推理失败。
- 原场景资源只读对比：RHSCI `assets-20261016/assets-manifest.json` 中本次场景 `102344022` 加全部 vMaterials 文件共 6014 项，与当前场景/材质的文件名集合和逐文件 SHA256 完全一致，缺少/多出/内容不同均为 0；结果 `original-scene-comparison.json`。没有替换资源或额外做资源 A/B，不将该范围外的物体/机器人资产声称为本次已比对。对已观察到的本样本卡顿，直接机制定位为模型块接续目标回跳，未确认需要修改的实现错误，不通过改变评测策略消除现象。
- 本轮诊断收尾正式阶段统计（level-0/1/2/hard）：STATE 成功 `0/0/0/0`、已完成 `7/8/7/8`；VGQ-AE 成功 `0/1/1/0`、已完成 `8/8/9/7`。仍未跑满每模型 200 次。opt-in trace 与分析器经三份真实轨迹检查，额外编译对照实际运行完成；四个 Python 文件 Ruff lint/format 通过，多 episode trace 在 Kit 导入前拒绝以免混淆步号。

- 最新视频/统计约定已写入 AGENTS 与评测入口：每模型、每难度保留前两轮全部 episode 左右目视频；之后不录像，不删除已有文件。当前旧 worker 无需重启，后续每轮客户端依据 repeat-NN 目录落实规则；新 worker 显式记录轮次和 video_rounds=2。结果按 level-0/1/2/hard 分别汇报成功数与已完成数。
- 本次复查阶段快照两模型均完成 `4/4/4/4`；STATE 成功 `0/0/0/0`，VGQ-AE 成功 `0/0/1/0`。已有 64 份带 success/failed 后缀的左右目 MP4；成功视频为 VGQ-AE/level-2/episode-3，640×360、50fps、1077 帧、21.54s。成功/失败各一份完整 FFmpeg 解码通过，成功视频 4 帧拼图 `logs/pi05-success-video-review.png` 已人工检查。实际日志未见新 USD/纹理错误、traceback 或 OOM，仍有官方篮子空碰撞 mesh 告警与 torchvision 弃用警告。核对了 30 步响应、队列耗尽后重预测、reset 清队列及原生成功判定，不将低成功率直接判作程序 bug。
- 后续逐份检查已关闭的 66 个 MP4，ffprobe 均返回有效 640×360、帧数和时长，与对应官方结果及左右目标签一致；此时 VGQ-AE 已完成 `4/4/5/4`、成功仍 `0/0/1/0`。录像规则 1–5 轮与显式轮次覆盖、重复/越界/损坏统计拒绝均经 CPU 检查。等待队列存在“worker 意外死亡但缺完成标记时一直等待”的静态/CPU 可复现缺口，未改动故障处理，详见错误记录；当前八路仍在运行。
- 收尾快照：STATE 成功 `0/0/0/0`、已完成 `4/5/4/5`；VGQ-AE 成功 `0/0/1/0`、已完成 `4/4/5/4`，均为第一轮阶段结果。四个修改/新增 Python 文件通过 Ruff lint/format；使用 mock 官方接口验证旧 worker 的 repeat-03 客户端实际关闭录像并在导入 Kit 前记录有效配置，没有额外启动 GPU 评测。

完整配置、来源和命令见 [实验说明](../experiments/pi05_between/README.md)，已在 `AGENTS.md` 登记为必读。

- 用户指定 STATE 与 VGQ-AE，各 checkpoint-40000，预测 30 / 执行 30，无 RTC；四难度各 10 个初态重复 5 轮，每模型 200 次，总计 400 次。重复不是 200 个独立初态。
- 模型 HF revision `59a1ea14d4346496d4e55c9fa58ea2f3330fd298`；上述两份推理权重已下载并 SHA256 验证。用户最新要求只下载已指定评测权重，其余全量下载调度、自动重试器和无关传输均已停止，保留已下载内容和分片，不自动恢复。
- 用户指定 dataset revision `ea9d480ff1d95f34b63d7ea368b1bc72e353fd57`；eval-data 共 151 文件、8,045,136,314 字节已全部验证。实际初态用最新官方 `SIMPLE-eval/fb67cd9a309cb4f3ce8f9470fecb6fa6d82c60d4`，2993 文件已验证，与 RHSCI 镜像一致。
- 匹配的两桌训练数据已下载、校验并解包，用于原处理器和统计；实际 fingerprint `b16d2e98348482834e282f2aa39ecc9154a60a6d6da4dad703f9ba9cc6141e2e` 与两份权重契约均匹配。不能用新版训练统计替换它。tokenizer SHA256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` 已验证。
- 已执行启动入口 `bash experiments/pi05_between/launch_8gpu.sh`。GPU 0–3 为 STATE 四难度，4–7 为 VGQ-AE 四难度；端口 8120–8127，DDS domain 100–107。首次尝试因 USD 诊断发现失效引用而主动停止；输出完整保留于同名 `-before-usd-repair` 目录，不计入正式 400 次。四难度渲染通过后已按相同命令重启，尚不能据此宣称完成物理评测。
- 输出根 `<运行目录>/evals/pi05-between-30x30-5cycles-20261104`。每卡有 `worker.json`、`server.log`；每轮有实际 `command.sh`、`console.log`、统计和视频。每卡完整 50 次后才产生 `COMPLETE.json`。
- 本机 keepalive 用 `/root/bin/xiancun pause 360` 暂停并释放 GPU；不要直接杀其 worker，监督器会重启。未停止其他生产训练。

## 已授权的后续评测队列

- 当前新准备 PID 419978、等待队列 PID 420186；日志分别为 `logs/followup-preparation-selected.log`、`logs/followup-queue-selected.log`。实际检查仅烤箱两个下载进程存活，准备进程等待原续传、并行核验本地水龙头 STATE；队列等待当前 400 次和所选资产。代码语法、5 组选择范围、只允许 2 份下载及 worker horizon 参数检查通过。

- **最新范围覆盖本节旧计划：** 只继续下载 GR00T 烤箱 STATE / VGQ-AE，原下载 PID 267262 / 267263 保持续传。已停止旧准备 PID 267176、旧等待 PID 288997 和交接 VGQ-AE 下载 PID 376533，保留所有文件/分片，当前八路评测未停。其他指定模型只评测已下载完整且本地 SHA256 通过的权重：π0.5 NO-STATE、GR00T 交接 STATE、水龙头 STATE。后续共 5 组、1000 次，交接/水龙头 VGQ-AE 暂缓；准备与队列采用 `*-selected.log`，实际命令见队列 README 最新段落。非烤箱归档缺失/损坏不自动补下载。
- 本次检查官方 `eval_stats.txt` 的阶段快照：STATE 四难度依次完成 3/4/3/4 次，均无成功，共 0/14；VGQ-AE 完成 4/3/4/3 次，level-2 成功 1 次，共 1/14。全部仍在 repeat-01，没有 `verified_results.json` 或完整模型结果。阶段样本不能代表最终 200 次成功率。

- 用户追加 7 组：π0.5 NO-STATE 两桌，及 GR00T 开烤箱/交接/开水龙头各 STATE、VGQ-AE；每组仍 5 轮、200 次，共新增 1400 次。先等当前 8 路均完成 50 次且逐轮核验结果，再按任务接续。
- 配置与来源、完整命令见已登记的 `experiments/simple_queue/README.md`：π0.5 30/30、40k；GR00T 16/16、20k，全部无 RTC。各任务使用自己的官方时限与成功判定，不套两桌参数。
- CPU 准备进程 `experiments/simple_queue/prepare.py` PID 267176，日志 `<运行目录>/logs/followup-preparation.log`。π0.5 NO-STATE 已完成下载/SHA256/解包及 horizon=30 核对，收据 `evals/followup-pi05-gr00t-5cycles-20261104/preparation/pi05-NO-STATE-between.json`。GR00T 交接 STATE 也已完整准备、horizon=16；水龙头 STATE 归档已下载/SHA256 通过。其余 4 份尚未全部就绪。用户限定只下载指定模型；全量调度 PID 86344、重试器 96854 及四个无关传输已结束，源码已取消自动恢复。
- **当前下载已恢复：** 用户最新明确专用节点不可用就换回、以能下载为优先。已将下载组切回香港 Y01，真实烤箱 STATE 权重 Range 请求返回 206、`bytes 0-65535/17701737683`、65536 字节，约 1.35 秒；再核对 PID 身份后 SIGCONT 准备进程 267176 和两个烤箱下载进程 267262、267263，复用分片/解包现场，无重复下载任务。GR00T 交接 STATE 已完成解包及 horizon=16 核对。只继续指定权重，当前评测及等待队列始终未暂停。
- 自动等待队列 `experiments/simple_queue/run.py` 已启动，日志 `<运行目录>/logs/followup-queue.log`；它等待当前真实 400 次及全部后续权重就绪。不会以进程退出码/health/预热代替 episode 完成。
- 队列每小时调用本机 `xiancun pause 360` 续暂停空闲占卡程序，防止长评测中六小时暂停到期；队列结束停止续期，原监督服务不删除、不禁用。
- GR00T 专用环境 Torch 2.7.0+cu128、Transformers 4.51.3、FlashAttention 2.7.4.post1 导入通过；后续真实 GPU 预热尚未执行，不能称已验证闭环。

## 材质检查当前状态

- 单卡双路独立容量探针已结束：GPU 2 / slot 1，端口 8130 / DDS 110，第二路模型预热通过；场景加载时总占用达到 30336 MiB、余量 1774 MiB，触发预设 2048 MiB 保护，仅停止额外 worker，未发生已观测到的 OOM。实际 episode 未完成，不计正式 400 次，也不能称双路稳定。结果 `evals/dual-gpu2-capacity-probe-20261104/result.json`；没有据此全卡加倍。原 8 路继续运行。
- 官方文件清单复核：50 ZIP、34068 文件缺少/多出/大小或 CRC 不符均为 0；88 处别名与官方 props/textures 内容一致。结果 `logs/material-inventory-audit.json`。**全部 40 初态实际 reset/render 已通过**：USD composition errors=0，80 处 table/table2 检查的 MDL 路径/导出名与官方 layout 一致；官方未绑定材质的 table2 仍未绑定。结果 `logs/pi05-between-material-rendering-all40/result.json`；日志 `logs/material-all40.log` 无缺失 USD/纹理/UsdToMdl/traceback。原生 close 已正常退出，诊断不计正式评测次数。

静态检查覆盖 40 初态、80 处材质使用、63 个 MDL 文件与导出名称、175 个纹理路径，未发现缺失。实际 USD 渲染另复现了 `/root/tmp` 失效旧软链接及缺失纹理，已检查/恢复 88 处 props/textures 别名。后续全 40 初态真实渲染和桌面绑定已通过，详见上列新收据；四初态旧收据保留溯源。后续正式 rollout 日志仍须检查；篮子碰撞 mesh 无 points 属官方资产已知限制，不归为材质问题，也未声称该物理资产无错误。

## 下载代理与流量约定

- Mihomo 配置 `/root/workspace/Haoxuan/mihomo/config/config.yaml` 已原生 `-t` 校验并通过本地 API 热加载；备份 `.before-download-route-20261104`。不要将代理密码、UUID、secret 或订阅链接复制进仓库/日志。
- “📥 下载专用” selector 现含香港 Y01、日本 Y05/Y06；**当前和重启默认均为香港 Y01**，按用户新授权优先保证可下载。HF `huggingface.co`/`hf.co`、PyPI、Torch、NVIDIA 下载域和 GitHub release/codeload 域继续走该组；普通“🔰 选择节点”仍为香港 Y01。x0.01 仅是日本节点标注，不代表已查实际账单。低倍率节点日后实际恢复再考虑切换；更新订阅可能覆盖本地规则，更新后重查。
- 复查节点：Y05 可解析但目标端口 TCP 超时；Y06 本机 DNS 返回 -2，不过 Google 与 Cloudflare DoH 均返回 NOERROR/3 条答案，解析得到的 A 地址直连其端口仍超时，所以不只是本机 DNS。两节点 HF 延迟测试分别返回 504/503，尚不能区分服务端离线或中间线路屏蔽，不猜改凭据或协议。香港 Y01 TCP 成功、HF 延迟 311ms、真实权重 Range 通过，已据此恢复下载。
- 恢复后连接审计已确认 `us.aws.cdn.hf.co` chains 为 `香港Y01 → 📥 下载专用`，且连接下载字节和烤箱权重分片数继续增加，不仅是进程处于运行状态。
- 已断开 23 条旧 HF 代理连接，避免切节点后旧 TCP 会话继续走香港 Y01。已下载分片全部保留。恢复前先用小 HEAD/Range 验证实际 HF CDN 路由，再核对下载进程身份、续传并检查 `/connections` 的 chains。
