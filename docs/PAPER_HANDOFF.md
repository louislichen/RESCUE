# RESCUE 论文研究交接

更新日期：2026-10-20，Asia/Shanghai。原实验工作区：`/home/wanghl/cl/RESCUE`。
实验代码与结果的已推送基准提交：`87b2c39`，分支 `main`，仓库
[louislichen/RESCUE](https://github.com/louislichen/RESCUE)。
最新拉取的远端实验归档提交：`9740a04`，新增 CSGB 4K 内部预实验；
其范围和结论见下面的同步记录，原 Qwen2 结果仍按各自协议解释。

这份文档交接已有讨论和证据，供 VSCode 新会话继续撰写论文。
它不表示完整方法已取得正结果，也不自动授权启动下一轮实验。
上面的绝对路径记录原实验环境；在本地写作时以本地仓库根目录为准，
直接使用本文的相对文件链接，无需建立相同的服务器路径或安装 GPU 实验环境。

## 0. 最新同步：执行更新与历史归档

### 2026-10-21：FP32 split-KV公平测速已完成，共同提速但相对10%目标未达

用户明确允许中间浮点末位不同，仍要求FP32 Target decode、关闭全局TF32、评测最终Token
与冻结原Target完全一致，并把相同执行实现给检索基线。只新增隔离执行候选，不改变非零
V-only Mapper、causal2/4策略、成本模型、原运行源码或暂停确认数据。

算子v1/v2各100/112后因“旧FSDPA参考不等于单键解析值”严格FAIL，两次原样保留；
v2清零Q/K没有修好该参考差异。另行预注册v3明确候选解析值严格、参考仅诊断，原112
输入/算法不变，真实112全部执行、16候选解析例逐位通过。此烟测不等于数值/Token/速度
等价。三版均有真实raw、独立可复算报告与图，不以新合同重分类旧失败。

随后48次真实模型门禁已exit0：两篇旧doc271/303×六长度×两Graph方法×原/候选，
全部输出匹配旧798封存canonical、48owner关闭；24对语义接受/工作轨迹及DraftQ/NativeS
相同、NativeT字节均不同，符合用户的有限Token合同。29344次候选Target层调用，672
fallback。31712项独审PASS、7报告测试PASS、五图实际视检、保存CLI summary字节重现，
local148文件完整SHA+size双屏障已核；[最终质量报告](../exp/round_overhead/out_fp32_splitkv_model_20261021_v1/analysis_v2/REPORT.md)。
这是两个复用文档的质量检查，不是生产速度/一般随机分布等价/新确认；它本身不含TPS收益。

为检验原六点曲线能否提高，另行冻结792请求公平计时，已完成H100隔离根
`/mnt/8T/cl/rescue_experiment_cache/fp32_splitkv_timing_20261021_v1`，tmux
`rescue_fp32_splitkv_timing_v1`，pane334007/模型PID334017。两方法均有原/候选执行，
24owner各冷271+原同32暖，共16旧文档×2重复×六长度；相同旧generate计时边界，全部
晚capture计费，构造/冷启动/close另报。77联合CPU及独审GO，15source/launch先验SHA通过。
2026-10-20 17:58:44 UTC正常exit0，actualtmux/PIDs/children缺席、GPU空，全部24prime/
768warm/24closed owner完整，792输出Token一致，384组原/新工作量与语义轨迹不变。
同新执行的RESCUE/检索六点收益为+6.120/+3.602/+1.977/+4.777/+1.557/−3.980%；
配对95%CI分别[0.526,12.365]/[−2.734,11.475]/[−3.081,8.001]/[0.389,10.003]/
[−2.444,6.067]/[−7.862,−0.301]%。自身加速8.006%–32.114%，基线也加速6.560%–
31.828%；共同执行提速不能充当相对算法优势。16K我方1462轮对1404轮，每完整generate
秒/轮近乎相同，账面上轮数差仍存在；无本轮stageprofile，不作单模块因果归因。
34份导出逐SHA/size双屏障一致、25891项独审/1293项独立数值对照、主线程12/12测试、
raw直接配对CI复算、九套最终图实际视检、保存CLI summary字节复现全过。162份产物已
0444可恢复只读封存，manifest SHA `9b727ca38bea86343dd6d5004342da2969ff3b6f1b4d605016445fdf967a47bd`。
[最终报告及九图](../exp/round_overhead/out_fp32_splitkv_timing_20261021_v1/analysis_v2/REPORT.md)。
[执行候选、真实失败保留、质量证据与计时协议入口](../exp/round_overhead/FP32_SPLITKV_CANDIDATE_V1.md)。
总体>10%目标保持active；本轮实验完成但六点>10门槛全部未达，不跨批次相减或外推完全
预热生产/SOTA，暂停的独立确认继续不动。下一隔离execution候选只准备M1非refresh
固定512位置维护：避免动态bool索引/两排序，保留同分规则及选位/Mapper/M3/Token不变；
还没有GPU或系统收益，不能承诺靠这一微优化补齐16K差距。

### 2026-10-20：六输入长度曲线已完整完成，16K出现负点

冻结Mapper／causal2/4策略／旧cost的六输入长度实测于北京时间18:16:11正常结束。
798新生成、140质量owner、18计时owner、96canonical单元、198tree raw完整；全部576
计时输出逐Token匹配。原始导出、独立raw审计、8套图实际审阅及保存源独立精确重跑均通过。
同16篇旧开发文档×2重复，实际2/4/6/8/12/16K对固定检索Top32/k4 Graph的TPS收益为
+5.8640/+5.0935/+2.8553/+5.6608/+1.7498/−3.7408%；点态95%配对CI仅2K、8K下界>0，
4K仍跨0、稳定优势未证实，16K也跨0、不能称显著落后。负点完整保留，不改参数重搜。
16K接受Draft/round1.815对1.930，轮数1462对1404，decode多6.379秒、outer多6.387秒，
prefill近乎相同；仅为账面分解，不能单独归因Mapper。

所有点相对固定完整树端口TPS均正（+80.66%到+38.10%），但它不是更强检索臂的替代。
主TPS含每owner一次cold后的32请求完整generate及所有晚capture；我方warm capture0，
检索16–22次/4.26–5.80秒，收益与有限摊销有关，不外推完全预热生产速度。三代表臂并非
新长度最优菜单；复用dev与非交错owner/硬件漂移限制保留，新确认数据未读，总体目标未达成。
[完成报告、八套图、raw、源码及核验](../exp/context_curve/out_h100_context_curve_v2_20261020/PROGRESS.md)。
最终只读封存清单SHA
`70e46d52c3f3f57b1efa8d1633b665f345d54d7656529c9b09e2d25e185bf41e`，
实际summary SHA `554cce558d3523ac96b414332095c9f01f39e3b5aeca86199aadab019d28602f`。
用户随后授权完成后通知；检查时实测已终态，因此直接收齐、核验和出图，无须新建或遗留重复监测。

### 2026-10-20：冻结同16篇检查已完成，六输入长度曲线另行补测

原130次请求于08:21:22 UTC正常exit0，实际模型进程与tmux均结束。64份质量请求、
64个质量owner和2个计时owner均核验关闭，130份输出逐Token符合Target；真实非零
V-only Mapper、causal2/4策略与旧cost保持不变。独立raw核查、六套图实际审阅、保存
入口独立重跑及2677份产物逐SHA/只读检查均已完成。最终清单SHA
`0f9e19fddd4cc57a6813c795b9ccdfc6f707bf816ec1323f94e4d45039db1c79`。

相对同规模最快检索Top32/k4 Graph：4K 47.8834 vs46.5915 TPS，+2.7728%，配对
文档95%CI[-2.4142%,8.9559%]；8K 38.6480 vs36.8361 TPS，+4.9190%，CI
[0.9825%,9.3774%]。全部12个保留对手点估计均正，仅满足旧开发可行性门槛。
4K稳定性未获证实，8K原单文档负差距在此队列未重现；不将跨批次变化解释为纯代码
提速。两臂有限服务轨迹的capture摊销不同，点态文档CI不覆盖owner/硬件漂移。
原新确认文档仍未打开；总体目标未完成。
[完整说明、raw、分析、图及封存证明](../exp/draft_graph/out_h100_frozen_devcheck_v1_20261020/PROGRESS.md)。

用户另明确要求输入上下文2K/4K/6K/8K/12K/16K曲线，不改Mapper/策略。
新独立协议只复用这16篇旧文档的自然文本、冻结三条代表臂，重新测量全部六长度计时；
不把代表臂称作各长度最优，也不替代原4K/8K全部12对手的最终确认。
六长度v1在首条AR前因完整上下文构造器误用Graph退出，新生成0次，原根/日志保留。
独立v2只把untimed AR显式设eager，Mapper/策略/计时方法/样本/统计不变；42项本地
回归全过（含真实tinyQwen AR），于09:10:30 UTC单次启动，PID3674374、tmux
`rescue_context_curve_v2`。09:10:52 UTC实际8项CUDA前检全过，新矩阵未完成，
现有两点图继续明确标PARTIAL。
[六长度实际状态、失败保留与冻结协议](../exp/context_curve/out_h100_context_curve_v2_20261020/PROGRESS.md)。

### 2026-10-20：不改Mapper／策略的同16篇匹配检查已单次启动

08:10:26 UTC（北京时间16:10:26）隔离根`frozen_devcheck_20261020_v1`单次启动，
精确tmux `rescue_frozen_devcheck_v1`、执行器PID3602661；两轮基线终态已先完整封存。
08:11:40 UTC只读确认同执行器live、实际模型PID3603253，8项真实CUDA前检全过，
已保存3/64质量请求及3/64关闭owner，尚无主计时结果。不是130矩阵已完成。
Mapper／causal2/4策略／旧cost／16旧文档／32请求顺序均冻结，保留全部12对手；
新确认Parquet、暂停71/320测试和新device-cap原型均不在本轮执行范围。
[真实启动、完整协议与后续结果入口](../exp/draft_graph/out_h100_frozen_devcheck_v1_20261020/PROGRESS.md)。

### 2026-10-20：32暖请求基线终态已完整封存；下一步为冻结我方旧开发检查

服务器记录2540请求于05:52:34 UTC正常exit0，root于07:40 UTC读回终态；实际PID与tmux
均已结束。32 AR、76冷prime、2432暖、76 owner、528树raw完整；2508份SD输出逐Token
匹配canonical。657原始导出SHA、全部76指标cell与短／长周期12对手并集经独立核对。
九套质量／TPS／捕获／周期／latency／GPU边界图已生成并由root实际查看；标题裁切
只在独立显示adapter修复，旧显示版本完整保留，冻结分析源码及summary／union逐字节不变。
保存源在全新目录精确重跑；最终只读清单SHA
`45f7d25a5516e8f1ffeaac51571941fce212d307837f7408a88841ccc60b0319`。
原183份程序源重新逐SHA核对不变；不把本地CUDA skips当作GPU通过。

本轮16篇旧开发文档×2重复，精确4K／8K、上限128；两个context最快均为Top32/k4 Graph，
46.5915／36.8361 TPS，开发点态95%文档CI为[43.7533,49.6620]／[34.9244,38.9590]。
旧短流候选不能删除，最终并集4K五项、8K七项（M12、匹配质量参考R3）。这是基线选择，
没有我方方法臂，不能与旧doc303数字交叉相减或宣称总体目标达成。
[终态说明、九套图、冻结分析与raw](../exp/draft_graph/out_h100_baseline_horizon_v1_20261020_final/PROGRESS.md)。

下一步仅原冻结130请求旧开发检查：不改真实非零Mapper／causal2／4策略／旧cost，
不含device-cap新原型、不读新的确认Parquet、不恢复71／320暂停测试。只有全部
质量／非零／不可变门槛及双context对全部12个对手TPS点均正，才考虑原先fresh16确认。
此条同步时我方130尚未启动；部署和真实启动凭据另记，不能用计划代替运行证据。

### 2026-10-20：长流选型正常运行；联合确认工具仅本地准备

04:13:26 UTC同一实际PID2874165仍live（elapsed1:51:49，Rl+）且tmux存在，
非原子计数32AR/43prime/1364warm、42个关闭owner，无FAILURE/PIPELINE_EXIT；8K继续。
04:06:29 UTC真实第三份快照完整覆盖4K全部38配置，各16篇旧文档×2；8K只有部分。
654份导出、152科学源锚点SHA通过；1311次SD逐Token匹配、264份树原始记录、39个
关闭owner及991次记录轨迹/工作对照通过，仍非完整矩阵终态或独立模型tensor证明。
九套PARTIAL图由root实际查看，保存源重跑summary逐字节一致，独立指标复算通过。
4K最高观测基线点为Graph Top32/k4：46.5915 TPS，描述性文档95%CI
[43.7533,49.6620]。8K两个展示cell分别16/7篇完整重复对，不作不等样本排名；
不冻结最终union，不与旧doc303的RESCUE数字交叉相减，也不宣称RESCUE胜出。
721份本地产物只读封存，清单SHA
`e1f04dd90d59758e642732a22d8806e58473dcbbadfdc40e2c892fecbfcd0405`，
状态仅PARTIAL_NOT_TERMINAL；不能作后续双终态冻结门控。
[第三份真实快照、独立审计、图、分析源和原始数据](../exp/draft_graph/out_h100_baseline_horizon_v1_20261020_snapshot_4k_complete/PROGRESS.md)。

另有**未推广的执行原型**：外部adapter只在第二个Draft Token合并Token与2／4 cap为
int64[2]一次host读回，不改模型、Mapper、特征或策略规则，默认关闭，也不并入冻结130
检查。root最终8项CPU差分全过（5760个决策/RNG组合及真实tinyQwen的KV/Q/选择/EOS/
catchup比较）；独立8项复跑也过，183份冻结程序源仍逐SHA相同。新增构建前覆盖测试后
封闭了类方法提前覆盖及@wraps伪装的边界洞；只改两个新原型文件，不改运行源码。
这不是实际GPU同步减少或提速证据，仍须真实Graph输出生命周期/逐步状态等价、独立
CUDA trace与未插桩配对端到端测试；stack/where/检查开销可能抵消收益。
[原型CPU核查及未满足的CUDA门槛](../exp/round_overhead/DEVICE_CAP_ADAPTER_REVIEW_v2.json)。

04:24:24 UTC原PID2874165/tmux仍live（elapsed2:02:48，Rl+），非原子32AR/47prime/
1498warm、46closed owners，最后8K retrieval32_k4_eager/warm25，无失败/终态标记。
未改变当前运行，不在计时服务器上并发跑Torch/模型预检。新增的CUDA语义预检和Kineto
边界解析工具只完成本地18项联测：14CPU通过、4CUDA明确skip；真实强制GPU入口在本地
缺CUDA分别exit2/1且不建输出，不能报GPU已过。语义预检包含实际Graph借用输出毒化
与逐bit KV/Q/动作比较，但这些body仍未执行；trace仅二步决策primitive，解析器只报保存
轨迹schema/linkage通过，不冒充独立硬件观察。原生8+1字节两回传与优化16字节一次
回传必须有实际GPU事件与唯一关联链，仍无E2E速度收益。
[新预检代码/测试与强制拒绝凭据](../exp/round_overhead/DEVICE_CAP_PRECHECK_REVIEW_v1.json)。

03:58:16 UTC同一实际PID2874165仍live（elapsed1:36:40，Rl+），32AR/37prime/
1162warm、36个关闭owner，无FAILURE/PIPELINE_EXIT，最后4K retrieval16_k1_graph/
warm9。4K菜单也尚未全部关闭；没有导出新快照、冻结完整union或启动后续。

冻结旧开发检查的新包装器、报告、独立接收/单次启动和只读收集工具现已完成本地实现。
全员冻结后444项完整回归426通过、18因无CUDA跳过；root看图后只改长标签换行及
非负数轴，再完整复跑444项仍426通过/18跳过，183份程序源前后SHA一致，152份
当前运行锚点（125本地+27封存树）也相同。最终本地凭证：
[444项日志](../exp/draft_graph/FROZEN_DEVCHECK_FULL_LOCAL_REGRESSION_v2.log)、
[183源码与152运行锚点](../exp/draft_graph/FROZEN_DEVCHECK_LOCAL_REGRESSION_RECEIPT_v2.json)。
这不是H100新实验完成；只有合成130编排/严格分析联调，六张布局检查图均明确标记
SYNTHETIC且root实际查看，不可作真实TPS结果。原444项v1凭证与旧423项凭证保留。
服务器仅只读查过固定Python3.10.20/依赖metadata、六资产路径均存在且无祖先/叶symlink。
真实stage1+PARTIAL stage2的新入口也在缺PIPELINE_EXIT时拒绝，未建输出或触发GPU：
[实际未完成门控凭证及早期诊断纠正](../exp/draft_graph/FROZEN_DEVCHECK_ACTUAL_PENDING_GATE_CHECK_v2.json)。
正式包/隔离根/新确认gate依然未建，无新候选访问、模型生成或旧源修改。

03:36:58 UTC同一实际PID2874165仍live（elapsed1:15:21，Rl+），32AR/29prime/
898warm、28个关闭owner，无FAILURE/PIPELINE_EXIT；未达到完整2540请求，不启动后续。
下一步新增的是**冻结旧开发可行性检查**，不是新算法搜索或最终确认：沿用全部16篇
已用文档和stage2原32暖序列，64个私有eager/Graph质量请求全部关闭后，两个新Graph
owner各付旧271冷prime再跑32暖，总计130次新生成、复用34个已封存AR参考。所有保留
对手都取完整stage2的同16篇/32暖计时，不能混入stage1原六暖数字。不同时间owner块
只作开发可行性与成本定位，不能作同期因果速度结论。预注册模板：
[冻结旧开发检查](../exp/draft_graph/frozen_devcheck_20261020_preregistration.json)。
只有质量/非零注入/不可变门槛通过且双context对全部保留对手的TPS点估计均正，才进入
原先不变的fresh16确认；否则保留fresh16未读，不换参数/文档/对手。描述性文档CI照报，
不冒称最终同时置信下界。此时包装器/报告/安全部署工具尚在实现，当前已完成本地检查
（见上方03:58更新），仍未打包、部署或启动。

03:30:21 UTC实际PID2874165（elapsed1:08:44，Rl+）与tmux仍live，32AR/25prime/
800warm、25个关闭owner，无FAILURE/PIPELINE_EXIT。当前仍是4K子矩阵，未完整结束、
未冻结最终对手，也没有新确认启动。目标保持active，不因工具绿或旧诊断完成而改称成功。

03:23:19 UTC同一实际PID2874165仍live，32AR/23prime/709warm、22个关闭owner，无失败。
补齐了独立v2一次性执行器/launcher及动态M/R收集、独立raw审计与封存工具，focused
13/6/9个CPU检查分别通过；全员冻结后root统一回归423项中405通过、18因无CUDA跳过，
175份程序源前后SHA相同，152份运行锚点也保持一致，不能用合成通过宣称GPU胜出。
新数据gate/实际计划/v1 capsule/v2执行根均未建立，两个v1入口仍硬禁用。共享测试
曾因并行改源触发10个稳定性拒绝，失败记录保留；详见
[未执行审查说明](../exp/draft_graph/UNEXECUTED_UNION_REVIEW.md)。

旧匹配doc303预实验的只读事后分解进一步明确8K差距：两个128-token请求的prefill
1.0160s对1.0171s几乎相同，decode5.6798s对4.9064s。RESCUE84轮对检索Top32/k4
Graph76轮（+10.5263%），平均整段decode每轮67.6168ms对64.5577ms（+4.7385%）；
1.157636=1.105263×1.047385只是记账恒等式，不是Target独占延迟、Mapper危害或可实现
优化收益的因果分解。样本仍为1篇复用开发文档/2计时重复，没有文档级95%CI，只示重复
范围；不与新16篇/32暖选型交叉相减。未改Mapper、长度、cost或当前正式协议。
共同Target/Draft内核优化也必须公平提供给基线。等输出时，相对TPS仅在我方省时比例
大于基线时改善；这个旧8K样本每outer秒的轮数和Draft步数反而都是基线更多，因此
“每轮/每步共同省固定时间”的简化模型不保证缩小差距，不能仅凭最大profile阶段许诺胜出。
一个尚未实施/测量的执行候选是第二步合并Token与2／4决策的host读回，严格保持RNG、
EOS、topk/argmax、全部Q/KV/features/cap不变；只可能少一次等待，不减少模型计算，
新增stack/copy也可能抵消。它不是已取得的优化结果，不能改当前冻结检查或冒充新算法收益。
三图（接受量/TPS/轮数成本）已实际查看，保存源重跑summary逐字节相等：
[事后诊断报告、图与源码](../exp/draft_graph/out_h100_match_pilot_cost_diagnostic_20261020_v3/REPORT.md)。
另一个旧coverage-Graph插桩运行（同doc303但不同于上面的未插桩匹配计时）在8K记录
Target verify约45.9ms/轮、Draft约4.3ms/步，支持把验证轮数与Target侧成本也纳入
未来优化定位。host/stream区间重叠且包括排队，不能相加当独立kernel成本或与上方未插桩
结果相减，不能推导保证可实现的加速上限。原始记录：
[旧四条profile](../exp/draft_graph/out_h100_coverage_pilot_v2_20261020/results/profiles.jsonl)、
[原插桩范围说明](../exp/draft_graph/out_h100_coverage_pilot_v2_20261020/results/analysis_v1/REPORT.md)。

02:59 UTC实际PID2874165仍正常运行，32AR/14prime/423warm、13个关闭owner，
不重启或改冻结协议。接收器P2边界收紧后，完整本地回归393项中375通过、18因无CUDA
跳过，165份程序源码前后SHA一致；此前386/392项凭证保留为历史检查，不覆盖。终态工具新增的第10项
测试通过生产API合成2540请求/76 owner/528树记录到分析、独立审计、封存、UNION验证
全链，仍不是实际GPU终态结果。新增隔离打包/接收工具只接受硬禁用胶囊，正式可执行
启动流程尚需另行审查。真实stage1完整+stage2 PARTIAL输入已验证在创建输出目录前
拒绝，无打包、部署、确认根或新候选读取。接收器现强制连续USTAR header，两个空
global-PAX/隐藏metadata纯内存复现均在建根前拒绝，不改科学源、协议或远程运行。

02:47:36 UTC只读快照新增真实完整树/后端证据：32AR/9prime/278warm/8closed owner，
654文件SHA通过。287SD逐Token匹配，33实际树原始记录、8完整owner账本、175次
冷/重复/同几何后端轨迹与工作计数比较通过，仍非全矩阵终态审计或全模型tensor证明。
同一32暖序列中，4K Top32 Graph k1/k4前6暖TPS为39.5044/38.7019，全部32为
39.9227/46.5915；k4相对k1+16.7044%，逐项文档配对区间[10.2131%,23.2228%]。
这是两基线探索性诊断，不是完整菜单赢家或RESCUE比较；前6/32文档覆盖不同，
不能把全部变化因果归于capture。保留全部短/长流候选并集的决策不变，不调我方参数。
九套PARTIAL主图与专项诊断已生成并实际查看，入口：
[第二份PARTIAL说明/图/数据/源码](../exp/draft_graph/out_h100_baseline_horizon_v1_20261020_snapshot_0250/PROGRESS.md)。

02:35:09 UTC只读核实：实际Python PID2874165仍在运行，已留存32AR/5prime/
154warm、4个关闭owner，无FAILURE；不重启、不修改当前冻结源码。
02:30:43 UTC快照留存653份文件SHA通过、32AR/3prime/88warm/2owners；
91个SD输出逐Token匹配AR，独立复算接受量/TPS/捕获计数通过。九套PARTIAL图已逐张查看，
k1仅8篇完整重复对、k12与resident k2各16篇，不能据不等样本选赢家；8K尚无计时。
显示适配只补实际样本计数，不改冻结分析源/原始数据。入口：
[PARTIAL说明与数据/图](../exp/draft_graph/out_h100_baseline_horizon_v1_20261020_snapshot_0234/PROGRESS.md)。

新建`fair_union_confirmation.py/data/report`，只做本地合成流程与CPU测试，未部署、
未读新确认数据。最终必须保留短流/长流选型的全部unique(context,mode)并集，M为8..16，
缺失匹配eager质量参考R为0..8；固定总请求162+34M+R，所有方法/两个context同32暖序列。
16篇文档为独立配对单位、两重复，seed20261026；普通95%区间与同时单侧95%下界分开，
正式所有M项点估计及Bonferroni-bootstrap下界(.05/M分位)均需>0，并满足严格exactness/非零注入。
这不是宣称基线选型或最终确认已经完成，工具测试也不是GPU速度证据。

**新数据规则尚未执行、会先于候选读取冻结**：原validation从索引0升序，首16篇自然正文
至少8192 Token、排除11份历史used/reserved元数据的文档SHA及精确4K/8K输入SHA，
也排除新cohort内部重复。不得按EOS/速度/注入结果换样本。原“索引>=900/长度>=16384”
仅未执行旧草案，已在任何候选读取前退役；调整依据是本阶段实际4K/8K输入范围，
不是观察候选性能后选样。新模板为TEMPLATE_NOT_EXECUTED；新旧确认launcher均硬exit64，
尚无最终source/data gate或服务器确认根。Mapper/旧cost/2／4策略仍冻结不搜索。

新联合确认工具已完成最终交叉审查与本地全量回归：386项中368通过、18项因
无CUDA跳过，前后162份source SHA一致；这不是服务器GPU验证或最终科学source freeze。
联测修复了新报告把排除元数据`rows`列表误认为整数的错误，独立核对11份cohort及
由其重建的denysets；失败日志保留，修正后MIN/MAX真实API合成链与standalone均通过。
终态收集/独立审计/封存工具已写成新文件，9项合成门槛检查通过；真实2540终态仍未到达，
不能把工具检查冒充已经完整审计。入口：[本地审查说明](../exp/draft_graph/UNEXECUTED_UNION_REVIEW.md)。

### 2026-10-20：用户确认本阶段最终验收范围

用户明确选择：**先以相同Qwen2权重的两种公平基线验收**，即完整树解码适配版
与最快检索版；本阶段不要求另行下载、训练或验证原论文Vicuna7B/68M。
不因此削弱正在运行的冻结基线选型：每context仍保留检索eager/resident/Graph
三个后端的开发优胜配置及完整树优胜配置，后续所有保留对手都须报告，防止短流
捕获摊销隐藏长流更有竞争力的后端。4K/8K必须同时通过预先冻结的吞吐与配对CI
门槛，并逐Token匹配同精度Target参考；Mapper继续保持真实非零。
当前538请求只是开发选型，未打开新确认文档，不能单凭开发winner宣布目标完成。
最终可以声明的范围限于此模型对、硬件、输入/输出协议和被公平选择的两类Qwen2
基线，不推广成原版Vicuna或普遍SOTA结果。此用户确认不改变已冻结的运行源码。

### 2026-10-20：538请求基线选型已完整核验；总目标尚未完成

服务器2026-10-20 02:11:56 UTC正常exit0，原tmux与Python PID2807824均已结束。
6AR/76cold prime/456warm、76owner完整，532个SD输出及EOS匹配独立AR；
112份树原始记录的overshoot/质量计数与整段计费一致。270份只读终态导出逐SHA通过，
含3份原协议已pinned的Target Tokenizer JSON，用于独立解析EOS151645。
独立审计重算所有76暖配置cell、完整排名与8个winner；七套PNG/SVG由root实际查看，
保存源码的重跑summary逐字节相等，全部产物已封存。未改旧服务器根或运行源码。
最终清单SHA `297657429434d6a3e058bde11e1ffc4890c0345382a436b2570c824dcaa39c10`；
本地`validate_baseline_freeze`返回`BASELINE_VERIFIED`。

| 输入 | 检索短流开发winner / TPS | 完整树短流开发winner / TPS |
|---|---|---|
| 4K | Top32/k1 Graph，38.4055 | Top32/nodes8/depth10，27.3159 |
| 8K | Top32/k8 Graph，35.0870 | Top32/nodes8/depth10，24.8536 |

仍保留4K Top32/k2 eager/resident、8K Top16/k2 eager/resident。主口径严格保留原6暖
SUM Token/SUM generate outer秒，含晚capture；不改成第二遍、服务摊销或扣除capture。
3篇复用train文档、2次重复，文档级10000次配对bootstrap，选择偏差与owner分块
时钟漂移仍需披露。这是两类Qwen公平基线的开发选择，**不是RESCUE最终超过它们**。

匹配32暖请求的另一个隔离开发补充已经本地实现并交叉审查，现已启动（见下）。
16篇固定旧文档/2重复/38配置/4K8K，共2540请求；同一32请求顺序用于所有配置，
每owner冷prime271且始终最多一个owner。源码证明并用真实CPU缓存回归测试确认：
非eager检索每owner最多63+k≤75种Graph状态，128条目上限足够，不必改容量。
运行器、数据与launcher已有31检查通过；最终报告含实际runner输出的2540模拟
fixture、文档配对区间、9套独立质量/速度/捕获/服务长度/漂移图及安全路径验证。
原8个候选与32暖新候选取并集，不能用后者替换/删除前者。无Mapper/策略/成本搜索，
不打开新的确认数据或暂停测试；未执行的旧fair_confirmation launcher继续硬禁用。

02:20:47 UTC只启动一次隔离根`/mnt/8T/cl/rescue_experiment_cache/baseline_horizon_20261020_v1`，
tmux`rescue_baseline_horizon_v1`；02:21只读核实RUN_STARTED及真实GPU前检正在运行，
尚不声称本轮正式科学结果。完整本地338检查为320通过、18因无CUDA跳过；
服务器`--validate-only`验证177份staged源码与538请求/76owner封存链成功。
传输SHA `723f39d663340fc4a44ea659d148a2517709f0d2830dd1443545948b084c0bf9`，
清单SHA `8a4dc1dafc41fcab8da6dd072e20c1d1324ab21ea362edb3b2470afd0146d658`。
所有运行源码只读，旧根与旧仓库不动，不重启、不省略配置或清理partial。
最新状态及部署日志入口：[RUNNING / PARTIAL](../exp/draft_graph/out_h100_baseline_horizon_v1_20261020/PROGRESS.md)。
02:23:04 UTC追加只读核实：服务器6组共64检查全部通过（4/15/9/10/16/10）；
实际运行Python PID2874165，15/32AR已保存、尚无暖请求，无FAILURE/PIPELINE_EXIT。
16固定输入、32个精确context SHA、152科学源码及旧data receipt均已冻结。正式计时
和完整配对图尚未就绪，不得引用预检速度为科学结果。后续只读监测该PID/tmux/日志，
不重复启动；终态后执行保存数据的独立审计、9套图、渲染检查、文字总结与封存。

结果：[说明与解释](../exp/draft_graph/out_h100_baseline_selection_v1_20261020_final/PROGRESS.md)、
[完整排名报告](../exp/draft_graph/out_h100_baseline_selection_v1_20261020_final/results/analysis_final_v1/REPORT.md)、
[接受质量](../exp/draft_graph/out_h100_baseline_selection_v1_20261020_final/results/analysis_final_v1/quality.png)、
[实际TPS](../exp/draft_graph/out_h100_baseline_selection_v1_20261020_final/results/analysis_final_v1/tps.png)、
[可运行分析](../exp/draft_graph/out_h100_baseline_selection_v1_20261020_final/results/analysis_final_v1/analysis_source.py)、
[原始暖请求](../exp/draft_graph/out_h100_baseline_selection_v1_20261020_final/results/runtime.jsonl)。

### 2026-10-20：重要新发现——捕获摊销的服务长度错配（以下为完成前历史记录）

01:47:58 UTC的只读快照已完整覆盖4K全部38配置；原538请求继续，8K尚未完整。
264份文件SHA通过，6AR/40prime/234warm/39closed owners，七套PNG/SVG已检查。
对Top32 Graph，六暖主口径k1=38.4055 TPS、k4=37.8141；各付2/16次新捕获，
0.372249/2.903610秒。完整repeat1三文档（全部纳入、均零新capture）却是
k1=39.4407、k4=44.6085，k4高13.10%。这只是原预注册允许的二级诊断，
不能事后改六暖指标、冒称完全稳态或推出32请求/新文档必胜。

**这改变下一步：不能用六暖每backend赢家直接验收32warm。** 保留三个后端也不能
解决同backend不同k的摊销反转。原538不改不重启，结束后仍完整分析/冻结原结果；
但本地新写的fair_confirmation.py/data/report与434+R模板仅是未执行候选，
没有最终source/data gate、没有创建服务器确认根、没有读取新确认内容。
`run_fair_confirmation_20261020_v1.sh`现已在任何远程状态操作之前硬exit64，
禁止把它当成可直接启动的正式确认。禁止因其CPU测试全绿而绕过这项方法学修正。

下一优先步骤：当前538完成后，另立全38配置、每context1旧271冷prime+32warm的
匹配服务长度开发补充；拟用已有旧train全12篇＋旧确认303/306/327/604共16篇
不同输入、每篇两次，每配置使用相同的冻结文档顺序。需先核实source identity、
文档SHA、4K/8K输入SHA、去重、自然长度和模板，失败不按性能换文档。
这是已反复使用的开发/训练cohort扩展，不是新的held-out；不读或重用暂停测试outcomes。
预计32新AR+76prime+2432warm=2540请求；尚未实现/部署/启动，不能把计划写成结果。
原8个短流choices和补充后choices均保留，最终报告所有unique retained对手，
不能按新确认结果选择较弱者。记录温度/时钟只能只读，不改GPU配置。

4组确认候选代码已覆盖真实非零Mapper、all-layer NativeKV/Q精确等价、独立quality与
timing owner、真实Graph128上限、EOS、资产/源码/图审链及文档同步bootstrap。
交叉审查修正了分析器错用不存在的own execution_backend与working_budget字段；
这些代码可复用，但最终协议和对手集合须在服务长度问题解决后重新审查。
当前没有任何最终超过两类Qwen2基线的确认结论，总目标保持active。

02:06 UTC更新：原538运行仍为实际Python PID2807824，已保存6AR/67prime/
402warm/67closed owners，无失败标记。此轮属于有效进展，不是因等待而判定blocked。
01:48快照已独立复算，保存源码重跑summary逐字节一致；289份本地产物封存为PARTIAL，
清单SHA `25b3ad050116bb0fa0ae7dd3577e638aa060590bc322c1bf7adbae434ae8e496`。

新`baseline_horizon`开发补充运行器与数据工具已经本地实现，尚未部署或启动。
源数据只读核实四文件SHA、12篇train身份211/213/220/226/233/236/246/248/
253/271/272/288，补充固定303/306/327/604；保存的16篇输入同prompt，均可自然
满足精确4K/8K。旧271既是各owner单独计费的冷prime，也是16篇warm之一，必须披露
这一同配置共享的重叠；不得把它算第17篇独立文档。原始corpus/shard身份继承旧数据
准备源码，不重新打开raw Parquet。所有样本都是已用过的开发数据。

本地已有30项检查通过：9项运行器（包括真实2540次控制流程模拟、76独立owner、
528份原始tree、失败先存raw再释放owner），16项纯stdlib数据验证，5项实际shell
`--validate-only`分支测试。CPU模拟不是CUDA或性能结果；最终分析器/跨文件整合审查
仍须完成。拟隔离根`baseline_horizon_20261020_v1`保持未创建状态；当前538必须先
完成原始数据、七套图和独立审计后才能通过新launcher门槛。

新入口：[预注册](../exp/draft_graph/baseline_horizon_20261020_preregistration.json)、
[运行器](../rescue/baseline_horizon.py)、[固定旧数据工具](../rescue/baseline_horizon_data.py)、
[启动脚本](../exp/draft_graph/run_baseline_horizon_20261020_v1.sh)。

入口：[最新PARTIAL说明/数据/图](../exp/draft_graph/out_h100_baseline_selection_v1_20261020_snapshot_0148/PROGRESS.md)、
[未执行确认运行器](../rescue/fair_confirmation.py)、[数据冻结工具](../rescue/fair_confirmation_data.py)、
[未执行确认分析器](../rescue/fair_confirmation_report.py)、
[候选模板](../exp/draft_graph/fair_confirmation_20261020_preregistration.json)。

### 2026-10-20：公平基线开发选型（已启动，未完成、未推送）

在前述成本重标定未获稳定增益后，冻结真实非零Mapper、旧cost与2/4策略，
先做**仅基线**的38配置选型，不再用开发结果改动我方参数，也不读取新的确认文档
或暂停71/320测试。2026-10-20 01:22 UTC只启动一次隔离根目录
`/mnt/8T/cl/rescue_experiment_cache/baseline_selection_20261020_v1`，
tmux `rescue_baseline_selection_v1`。启动时H100空闲；140份源码冻结并只读，
传输包SHA `70f62f5f973111c596242b15f04e1e96bfc460810042307dbc3d49e9c833e677`，
清单SHA `69ef185a3de2b0e8aec729cee08265d9e95c1b35963f1ada661326d17bd2330a`。
本地261项检查243通过、18因无CUDA跳过，完整538请求模拟和报告14项检查通过；
正式运行先经过独立真实GPU门槛，不能将本地跳过当作GPU验证。

固定复用train-tune文档271/272/288，精确4096/8192输入、128输出、两次暖重复。
检索Top16/32×k1/2/4/8/12×eager/resident/Graph共30配置；完整Qwen树
Top16/32×节点8/32/50/64共8配置，depth10、threshold0.7不变。
计划6AR、76独立冷prime、456暖请求，共538请求；每context/config一个私有owner，
依次构造、冷prime、六次暖请求、同步关闭，最多一个owner存活。所有合法晚捕获、
树overshoot都计费；主口径完整generate outer，不借正确性Graph或免费预热。
共同模型加载/共享wrapper准备等同排除，构造、冷启动与七请求服务摊销另报。

完整核验后，每context按六次暖请求池化Token/time，分别冻结检索三个执行后端及
完整树各一个winner，共八个后续确认候选。保留全部后端是为了防止短流捕获摊销
使长流可能有竞争力的后端被提前丢弃。后续我方必须对每context四个固定对手全部
比较，不能看确认结果再选择较弱基线。该38配置、3篇复用文档的选型有winner bias，
**本次即使完成也不构成独立确认、产线速度、原版Vicuna或超过SpecExtend的证明**。
服务器38项真实GPU前检全部通过，包括k12 Graph/EOS/catchup；已进入正式科学请求。
01:26:20 UTC只读快照留存6AR、4prime、22warm、3个完整owner，263份文件SHA通过；
本地七套PNG/SVG已检查，全部PARTIAL。8K仅有AR、没有计时，图中空白不是零TPS；
不冻结临时优胜配置。01:28 UTC确认tmux及实际Python PID2807824仍在运行，
6AR、7prime、37warm留存、6owner关闭，无FAILURE/PIPELINE_EXIT/COMPLETE；
GPU仅该PID，约40,578MiB。不要把锁文件或PARTIAL图当作运行结束的证明。
服务器不安装Matplotlib；完整结束后仍需拉回严格分析、七套图与文档配对CI，检查渲染后
才报告完成。部分失败保留原根目录，禁止重启或自动省略配置。当前尚无本次最终结论。

进度快照：[PARTIAL说明与图/原始数据入口](../exp/draft_graph/out_h100_baseline_selection_v1_20261020_snapshot_0126/PROGRESS.md)。
传输包SHA `22a99033a457df3d0cb67fe1ae6f25e9d7728b952ae22e404dae3c83e67a98a4`。
后续确认设计仅只读审查、未冻结或启动：可用预先排除历史/保留cohort后的首16篇
自然长validation新文档，两次重复，对每context四个冻结对手全部比较；不能在确认后
挑较弱对手。普通逐项95%文档配对CI与真正同时下界须区别，具体门槛先于新数据锁定。
Target post-attention MLP子图Graph也仅完成可行性审查，没有实现或额外速度结果，
不能声称完整verify可直接捕获，亦不能只给我方而不给保留基线同等实现与捕获计费。

入口：[预注册](../exp/draft_graph/baseline_selection_20261020_preregistration.json)、
[运行器](../rescue/baseline_selection.py)、[严格分析器](../rescue/baseline_selection_report.py)、
[隔离启动脚本](../exp/draft_graph/run_baseline_selection_20261020_v1.sh)。

### 2026-10-20：Draft Graph 单系数成本重标定（已核验，未推送）

仅重标定旧成本 JSON 的 `student_step`，不训练 Mapper/策略，也不重拟合
`verify_beta`、`fixed_round` 或 `cache_update`。本地237项测试中220通过、17因无CUDA
跳过，独立审查未发现P0/P1。隔离根目录
`/mnt/8T/cl/rescue_experiment_cache/draft_cost_recalibration_20261020_v2`，
tmux `rescue_draft_cost_recalibration_v2`，于2026-10-20 00:16:41 UTC启动一次，
00:24:53 UTC完成全部94请求、exit0，当前无该运行任务。
v1在纯CPU报告绘图前检因服务器缺Matplotlib失败，零科学请求；117条源码/日志SHA与
原失败根目录完整保留。v2不改共享依赖，仅将已测试的报告/绘图移到本地完成，科学协议不变。
原仓库、旧根目录、暂停71/320测试不变；不改VPN、不使用密码。

冻结94请求协议：原训练文档246的2AR＋2冷prime＋4暖profile；只用repeat0所有
逐步样本池化中位数拟合，repeat1在4K/8K分别检查20%同文重复性，所有暖profile必须
零新增捕获且匹配自身prime轨迹；先封存成本再读开发271/272/288。开发6AR、24质量、
8独立冷prime、48交错热计时，比较旧risk、新step成本risk、真实非零Mapper＋固定4
（关闭M3），以及同样Graph的Top32/k4检索。M1仍eager，Draft Graph，FP32 Target、
BF16 Draft/前缀、128输出、每文档两重复。所有合法热捕获计费，冷与active-service另报。

实测 `student_step` 13.444691ms→4.736646ms；全部86个SD请求输出均匹配独立AR的
128 Token。24个质量请求留存28层Native KV/每步Q、真实非零V证据，固定4关闭M3。
39项服务器前检通过，221条外层SHA通过；独立配对复算与保存源码的真实数据重跑一致，
五套PNG/SVG均已人工检查。246与三个开发文档均来自原train语料，不构成held-out。

| 新成本risk相对 | 4K TPS变化，95% CI | 8K TPS变化，95% CI |
|---|---:|---:|
| 旧cost risk | −1.0585% [−1.6817%, −0.2994%] | +1.9028% [−1.7456%, +7.9432%] |
| 同Mapper固定4 | −2.4461% [−6.0426%, −0.4294%] | +0.0878% [−0.7732%, +0.5769%] |
| Top32/k4 Graph检索锚点 | +11.9843% [+1.8607%, +18.2245%] | +16.4396% [+5.1909%, +30.3105%] |

**单系数重标定没有稳定增益，不提升为默认。** 接受长度虽从1.6770/2.1571升至
1.6959/2.2757，cap4更接近固定4，却未稳定提速。检索锚点暖请求仍支付16/13个新
Graph捕获、2.3541/1.9329秒，RESCUE为零；正差距是有限请求流的真实捕获计费结果，
不是全部形状预热后的稳态或最强SpecExtend胜出。先前doc303的8K负结果不被覆盖。
保留已验证的Draft Graph优化，冻结Mapper，不借本轮开发再调cost；后续应先公平选择
检索执行后端/固定长度与完整树节点配置，再打开新确认文档。旧71/320仍暂停，总目标未达成。

结果入口：[解释与决定](../exp/draft_graph/out_h100_cost_recalibration_v2_20261020/results/analysis_v1/INTERPRETATION.md)、
[自动严格报告](../exp/draft_graph/out_h100_cost_recalibration_v2_20261020/results/analysis_v1/REPORT.md)、
[接受量](../exp/draft_graph/out_h100_cost_recalibration_v2_20261020/results/analysis_v1/acceptance.png)、
[TPS](../exp/draft_graph/out_h100_cost_recalibration_v2_20261020/results/analysis_v1/tps.png)、
[捕获/标定](../exp/draft_graph/out_h100_cost_recalibration_v2_20261020/results/analysis_v1/calibration_capture.png)、
[原始账本](../exp/draft_graph/out_h100_cost_recalibration_v2_20261020/results/runtime.jsonl)、
[分析源码](../exp/draft_graph/out_h100_cost_recalibration_v2_20261020/results/analysis_v1/analysis_source.py)。
入口：[预注册](../exp/draft_graph/cost_recalibration_20261020_preregistration.json)、
[运行器](../rescue/draft_cost_recalibration.py)、[分析器](../rescue/draft_cost_recalibration_report.py)、
[隔离启动脚本](../exp/draft_graph/run_cost_recalibration_20261020_v2.sh)、
[v1零请求失败记录](../exp/draft_graph/out_h100_cost_preflight_failed_v1_20261020/REPORT.md)。

### 2026-10-20：M1 不变贪心循环 Graph 试验（本地新增，未推送）

在 Draft 已启用 Graph 的前提下，只把 M1 的28轮贪心覆盖内循环改为实际pool
形状的Graph重放；粗选、归一化、选位目标、非零Mapper、2/4策略和旧成本均冻结。
v1先在CUDA测试门槛失败（Python bool索引写入不可捕获），**零个科学请求**；
原失败根目录/源码/日志完整保留。独立v2只把Graph中的布尔used-mask写入改为
`index_fill_`，不改浮点运算或eager参考；40项真实H100测试全部通过。

v2在复用开发文档303、精确4K/8K、128输出上完成22请求：2AR、4严格正确性、
4独立冷prime、8无profile交错热计时、4单独暖profile，四个持久私有owner。

| 输入 | 原Draft Graph热TPS | 加M1 Graph热TPS | 热变化 | 冷变化 | 3请求计费吞吐变化 |
|---|---:|---:|---:|---:|---:|
| 4096 | 47.8716 | 48.6920 | +1.7137% | -14.3613% | -5.4087% |
| 8192 | 37.9337 | 38.6607 | +1.9165% | -14.3315% | -5.0705% |

20次SD均与独立AR的128 Token一致；224层缓存对、686条Q、逐轮选位、12维特征、
cap和接受轨迹完全一致，Mapper确有非零校正。8次热计时先于所有profile；
profile时间/Token从主TPS和active-service中排除；关闭后所有Graph entries清零。
接受长度不变。228条SHA（连清单229文件）、原始账本、三套图已独立核验；
保存的分析脚本及helper可脱离仓库重跑。n=1，不给文档CI，不是SpecExtend胜出。

**该微优化只改善热吞吐，冷与三请求计费口径反而更慢，因此不提升为默认设置。**
profile中Target验证host约1.50/1.93s，明显大于M1选择的约0.126/0.128s；
阶段host/stream不是独立可加的纯kernel成本，不能据此断言全是CPU launch瓶颈。
另外发现旧长度控制器仍按Draft单步13.445ms估价，而新Graph profile约4.32–4.33ms。
下一优先候选是**仅用原训练文档246重标定成本**，不从开发文档303拟合参数；
尚未启动该标定。Target dense区部分Graph也仅完成只读设计，未实现/测速，
且其共用执行优化同样应给检索基线使用。旧71/320测试仍暂停，正式目标未完成。

入口：[报告](../exp/draft_graph/out_h100_coverage_pilot_v2_20261020/results/analysis_v1/REPORT.md)、
[接受量](../exp/draft_graph/out_h100_coverage_pilot_v2_20261020/results/analysis_v1/acceptance.png)、
[TPS](../exp/draft_graph/out_h100_coverage_pilot_v2_20261020/results/analysis_v1/tps.png)、
[机制诊断](../exp/draft_graph/out_h100_coverage_pilot_v2_20261020/results/analysis_v1/diagnostics.png)、
[原始测速](../exp/draft_graph/out_h100_coverage_pilot_v2_20261020/results/runtime.jsonl)、
[原始profile](../exp/draft_graph/out_h100_coverage_pilot_v2_20261020/results/profiles.jsonl)、
[分析源码](../exp/draft_graph/out_h100_coverage_pilot_v2_20261020/results/analysis_v1/analysis_source.py)、
[核验](../exp/draft_graph/out_h100_coverage_pilot_v2_20261020/results/analysis_v1/VERIFIED_COMPLETE.json)、
[v1零请求失败前检](../exp/draft_graph/out_h100_coverage_gpu_gate_failed_v1_20261020/REPORT.md)。

### 2026-10-20：相同 Graph 优化后的 SpecExtend 差距 pilot（本地新增，未推送）

在相同 H100、Qwen2 权重、FP32 Target/BF16 Draft 与前缀、128 输出下，
只用复用开发文档303的精确4K/8K、两次热计时，完成44请求：2 AR、12正确性、
10独立冷prime、20热计时。固定真实非零Mapper与2/4策略/成本；检索控制也获得
持久缓存和eager/resident/Graph三后端。完整Qwen树控制保留历史n8、depth10、
Flash prefill和hybrid attention，官方Flash2.8.3仅装入新隔离venv。

| 输入 | RESCUE Graph TPS | 同样Graph的Top32/k4检索TPS | 相对变化 | 完整树n8 TPS |
|---|---:|---:|---:|---:|
| 4096 | 48.0804 | 45.8394 | +4.8887% | 27.3849 |
| 8192 | 38.2277 | 43.2102 | -11.5307% | 23.3977 |

这是每个模式两次完整generate outer的Token/time池化，不含共同模型加载。
所有冷/热捕获成本保留；冷启动和构造+冷prime+两次热请求+关闭的active-service
另报，不能用后者替代稳态口径。完整树额外计算2/3个Token的耗时全部计入，
其接受量为含overshoot的raw每轮计数，不伪造树接受率或未返回的尾部Token ID。

42 SD输出均匹配独立AR的128 Token；相同算法各执行后端的选位、决策、
560个质量层缓存记录及1781条Q记录一致，且M2保持真实非零。43项H100测试全过；
245个证据清单条目（加清单自身246文件）、原始账本和三套图均已独立核验。
**n=1，不给文档CI；k4/n8是历史开发配置，并未重新搜索最优基线。因此不能据此
宣布超过SpecExtend，尤其8K仍慢11.53%。正式目标保持未完成，旧71/320测试仍暂停。**
下一候选仅将M1不变的贪心覆盖内循环做Graph重放，默认关闭；尚无该候选速度结果。

入口：[报告](../exp/draft_graph/out_h100_match_pilot_v1_20261020/results/analysis_v1/REPORT.md)、
[接受量](../exp/draft_graph/out_h100_match_pilot_v1_20261020/results/analysis_v1/quality.png)、
[端到端TPS](../exp/draft_graph/out_h100_match_pilot_v1_20261020/results/analysis_v1/tps.png)、
[延迟/捕获](../exp/draft_graph/out_h100_match_pilot_v1_20261020/results/analysis_v1/latency_captures.png)、
[原始计时](../exp/draft_graph/out_h100_match_pilot_v1_20261020/results/runtime.jsonl)、
[分析源码](../exp/draft_graph/out_h100_match_pilot_v1_20261020/results/analysis_v1/analysis_source.py)、
[核验](../exp/draft_graph/out_h100_match_pilot_v1_20261020/results/analysis_v1/VERIFIED_COMPLETE.json)。

### 2026-10-20：持久 Draft Graph 端到端执行优化（本地新增，未推送）

前一日单文档 32 输出 pilot 之后，按预注册完成六篇复用开发文档
303/306/327/604/613/614、精确 4096/8192 输入、128 输出、三次计时重复。
12 独立 AR＋36 严格正确性＋6 独立冷 prime＋108 无 observer/profile 的交错计时，
共 **162 请求**，均生成 128 Token 并匹配 Target。每 context/mode 使用独立私有
持久缓存/Graph owner，冷 prime 固定为文档303，不借正确性阶段的 Graph。
三模式共享冻结权重，但每请求显式 install/restore Target hook，切换成本纳入 outer。
所有热 Graph 请求零新增捕获、owner/缓存地址不变，六类实际形状均由冷 prime 覆盖。

| 输入 | GPU 常驻相对原优化 eager TPS，95% CI | 常驻＋Graph 相对 eager TPS，95% CI |
|---|---|---|
| 4096 | +5.5327% [+5.4123%, +5.6496%] | **+59.6553% [+58.8655%, +60.5639%]** |
| 8192 | +4.6869% [+4.5858%, +4.7910%] | **+47.4942% [+47.1129%, +47.9420%]** |

主口径为完整 generate outer：含 prefill、缓存重置、hook 切换、decode、请求清理，
各模式均不含共同模型加载。每文档先池化三次实际 Token/time，再对六文档配对差异
做 10,000 次 bootstrap；不能把 108 请求或两个 context 算成独立样本。
4K 的 pooled warm TPS eager/resident/Graph 为 29.0267/30.6343/46.3322，
8K 为 26.5729/27.8179/39.1913；池化比值与上表逐文档配对百分比不是同一统计量。

首次 128-token 请求也包含捕获成本：文档303上 Graph 冷 TPS 在 4K 为35.8549
对30.3206（+18.25%），8K 为29.9329对25.7823（+16.10%）；捕获约818/830ms。
冷请求每 context 仅一个文档，不给文档级 CI。这不覆盖或否定旧32输出 pilot 的
冷启动减速。计入构造＋一次冷 prime＋18次热请求＋关闭的实测 active-service sum，
Graph TPS 为45.6202/38.5548，相对 eager 为+56.81%/+45.33%；因模式交错，
这不是连续单模式 session wall 或生产服务吞吐。真实 context wall 另行保存。

接受量、逐轮决策与输出不变。独立质量审计确认所有28层 Target/Draft K/V 和
6294条 Q 记录哈希、选位、有效12维特征、draft/check/接受轨迹一致，且 Mapper
确有非零校正（并非关闭 M2）；最后 remaining=1 无控制器特征是合法边界。
服务器26项相关测试全过，含六项真实 CUDA Graph 测试；本地156项中148过、
8项因无CUDA跳过。320份冻结证据 SHA、本地独立原始计时重算与六张图已核验。
原服务器仓库、旧实验和71/320暂停测试未修改，未训练、改成本或搜索新策略。

**这次通过的是完整配置的执行速度门槛，不是三模块算法质量或超过 SpecExtend 的门槛。**
它仍是 H100、六篇复用开发文档、单模型对、固定非零 V-only Mapper 与2/4策略，
不是4090、独立测试泛化、原版Vicuna树解码或真实边云网络结果。默认仍为 eager，
需显式启用 `draft_execution='cuda_graph'` 并复用同一个 Engine。记录仅在本地，未推送。

入口：[最终报告](../exp/draft_graph/out_h100_benchmark_v1_20261020/results/analysis_final_v2/REPORT.md)、
[接受长度](../exp/draft_graph/out_h100_benchmark_v1_20261020/results/analysis_final_v2/acceptance.png)、
[端到端TPS](../exp/draft_graph/out_h100_benchmark_v1_20261020/results/analysis_final_v2/warm_tps.png)、
[延迟](../exp/draft_graph/out_h100_benchmark_v1_20261020/results/analysis_final_v2/stage_latency.png)、
[冷捕获/摊销](../exp/draft_graph/out_h100_benchmark_v1_20261020/results/analysis_final_v2/cold_and_accounted_service.png)、
[原始计时](../exp/draft_graph/out_h100_benchmark_v1_20261020/results/runtime.jsonl)、
[分析源码](../exp/draft_graph/out_h100_benchmark_v1_20261020/results/analysis_final_v2/analysis_source.py)、
[预注册](../exp/draft_graph/benchmark_20261020_preregistration.json)。初版图留在 `results/analysis`；
final v2 只修正诊断图的颜色/标签，未改原始数据或统计。执行时保留旧冻结report源码，
后续本地分析源码另存；不把报告排版更新伪装成已经测过的运行代码。

### 2026-10-19：Draft GPU 常驻与单 Token Graph 小验证（本地新增，未推送）

实现默认关闭的 `resident` / `cuda_graph` 执行路径，只捕获单 Token 模型计算主体，
不捕获整轮生成循环。GPU Token/位置和 Mapper shadow binding 复用；Native KV 写入、
逻辑长度、逐 Token EOS 和第二 Token 后的 2/4 决策仍在 Graph 外。
Graph 使用真实工作缓存长度，不用额外 padding 改变注意力。
缓存重新分配会使 Graph owner 失效；只有同一个 Engine/缓存复用才复用 Graph。
这不是完全消除 CPU 控制与逐 Token 同步。

对照为下面三项开销优化实验的 **after** 源码，不是裸 `24f63f9`、legacy 路径或
SpecExtend。真实非零 V-only Mapper、已训练 StepBudget/成本及算法配置全部冻结。
H100 80GB 上使用复用开发文档 303、精确 4096 输入、32 输出、两次计时重复：
1 次 AR、3 次正确性、12 次主计时，共 16 请求。六个计时 session 各建新 Engine，
先冷后暖，未借用正确性阶段的 Graph。模型加载不计入请求，各模式一致。

| 执行模式 | 冷请求 TPS | 热请求 TPS | 完整双请求会话 TPS |
|---|---:|---:|---:|
| 原优化 eager | 20.0421 | 20.3447 | 19.8601 |
| GPU 常驻主体 | 21.0395 | 21.2691 | 20.7713 |
| 常驻＋CUDA Graph | 16.9379 | 29.2098 | 20.9815 |

常驻版冷/热分别 **+4.98%/+4.54%**；Graph 冷/热分别 **-15.49%/+43.57%**，
平均首次捕获成本 **0.779 秒**。完整双请求 session 分别 **+4.59%/+5.65%**，
Graph 比常驻仅多 **1.01%**；session 包含中间记录 I/O、安全检查和关闭开销，
不能冒充纯产线服务吞吐。不能只报道热请求的 43.57% 而隐藏首次捕获成本。

三路每请求均接受 18/提出 53 个 Draft Token、15 轮、56 次 Draft 前向，
平均接受长度 1.2 Token/轮；质量没有提升也没有改变。Target 输出、选择位置、
12 维特征、2/4/接受轨迹、全部有效 Native KV 和每步 Query 的字节哈希一致，
均观察到真实非零 Mapper 注入。服务器 19 项测试通过，含五项真实 CUDA Graph 测试；
150 份证据哈希及原始计时由独立审查重算，四张 PNG/SVG 已人工检查。

**COMPLETE 仅指这个小规模 pilot 的运行、核验、作图和报告完整。** n=1 文档，
两次 timing repeat 不是独立样本，没有文档级 95% CI，不支持 8K、128 输出、
4090、独立泛化或优于 SpecExtend 的结论。下一步可验证持久 Engine 下捕获摊销，
但本记录没有自动启动更大实验。原服务器仓库、暂停测试及旧实验均未修改。

入口：[报告](../exp/draft_graph/out_h100_v1_20261019/pilot/analysis/REPORT.md)、
[接受长度图](../exp/draft_graph/out_h100_v1_20261019/pilot/analysis/acceptance.png)、
[冷/热 TPS 图](../exp/draft_graph/out_h100_v1_20261019/pilot/analysis/cold_warm_tps.png)、
[会话 TPS 图](../exp/draft_graph/out_h100_v1_20261019/pilot/analysis/session_tps.png)、
[延迟图](../exp/draft_graph/out_h100_v1_20261019/pilot/analysis/latency.png)、
[原始计时](../exp/draft_graph/out_h100_v1_20261019/pilot/runtime.jsonl)、
[分析源码](../exp/draft_graph/out_h100_v1_20261019/pilot/analysis/analysis_source.py)、
[执行实现](../rescue/draft_graph.py)。执行时冻结的 report 源码与后续本地分析源码
分别保存：后者只修正图中文字排版，不改原始数据或远端冻结文件。

### 2026-10-19：H100 三项执行开销优化验证（本地新增，未推送）

用户授权在 H100 上验证执行优化；原仓库和暂停的 71/320 测试均未修改。
对照为提交 `24f63f987f36fe167620e2a418d57574860cf7df` 的已优化执行路径，
不是 `legacy-execution`。处理仅包括不读取无用长度评分、缓存固定结果几何、
复用归一化注意力 head mean；非零 `full_final.pt` Mapper、已训练 StepBudget、
成本和全部算法设置冻结。实际长度仍在提出两个草稿后决定总长 2/4，容量 12
不是十二 token 动作。基础模型沿用 Qwen2 7B/1.5B，Target FP32、Draft/prefill BF16。

在六篇复用开发文档 303/306/327/604/613/614 上使用精确 4096/8192 输入、
最大输出 128、每条件三次重复；独立预热四次、AR 参考十二次、正确性二十四次、
无 observer/profile 主计时七十二次，共 112 请求完成。选中位置、12 维因果特征、
draft/check token、长度、接受轨迹和最终 Target 输出全部相同，且每个正确性请求
均有实际非零注入。服务器 26 项相关测试通过，包含两项 CUDA 检查。

整请求 TPS 的逐文档配对均值变化：4K **+0.1671%**，95% CI
**[-0.0485%, +0.3961%]**；8K **+0.1557%**，CI **[+0.0504%, +0.2648%]**。
接受长度差为零。四张 PNG/SVG 图、原始记录、冻结源码/权重哈希和分析源码均已保存，
独立重算及 141 个证据文件 SHA 核验通过。**COMPLETE 指运行和报告完整，不指加速成功**：
两种上下文区间下界均为正的速度门槛未通过；不能写成明显系统加速、4090 结果、
独立泛化或超过 SpecExtend/native SD。本轮没有重训、调参或新增外部 baseline 对照。

入口：[完整报告](../exp/round_overhead/out_h100_v1_20261019/results/analysis_final/REPORT.md)、
[统计](../exp/round_overhead/out_h100_v1_20261019/results/analysis_final/summary.json)、
[原始主计时](../exp/round_overhead/out_h100_v1_20261019/results/runtime.jsonl)、
[执行 Runner](../rescue/round_overhead_run.py)。后续分析源码/说明更新与执行时冻结副本
分开保存，未覆盖任何原始记录。这一批仅在本地，未提交或推送 GitHub。

### 2026-10-17：远端归档与系统实现

已从 `origin/main` 快进拉取 `960f1be → 9740a04`。新增
[CSGB 归档入口](../exp/specextend_bridgekv/csgb_acceptance_pilot_v3/README.md)，
包含执行时冻结源码、控制器参数、原始请求、分析和六组结果图。
它是与下面 Qwen2 路线分开的 Vicuna 内部预实验，没有更新覆盖选择算法的有效性证据。

### 这次实际验证了什么

- Target / Draft：Vicuna 7B 16K / Vicuna 68M；输入 4096，最大输出 128 token。
- 工作缓存沿用 Attention Top32、chunk 长度 32 token，每四轮刷新。
- 使用 Native V，mapper 关闭，树最大深度固定为 10；动作是整棵候选树节点预算
  `{8,16,24,32}`。**节点总数与一次草稿的最大深度/生成长度不是同一个量。**
- 从上一轮验证的 Target 注意力、下一轮 Root 实际可读位置和缓存构造时注意力
  快照，计算覆盖率 C、相对缓存锚点的注意力变化 D，输入为 `[1,C,D,C×D]`。
- 使用 softmax 线性策略和完整请求奖励 `N − rho×T`，仅在请求结束后更新。
  该归档的实际控制器不是同目录历史工具中的 LinUCB。
- 历史文档 0–15 用于训练，16–23 是训练内监测文档；未打开独立开发/确认集。
  一个训练种子 101，三个采样动作流，每个动作流两个计时重复；统计单位是八篇文档。
- 每文档先对同动作流两个计时取中位数，再平均三个流，最后做文档配对 bootstrap。
  三个流不是三个独立训练种子，也不能把 336 个监测请求当成 336 个独立样本。

### 已记录的结果

| 指标 | 相对训练集选出的 fixed8_native | 文档配对 95% CI |
|---|---:|---:|
| 平均每轮接受 Draft token 数 | +0.3013 | [+0.1684,+0.4070] |
| 完整请求 TPS | -2.35% | [-4.56%,-0.79%] |

接受量的绝对均值为 fixed8 1.7909、CSGB 2.0922；这是 token/轮，不是接受率百分点。
常量输入策略也有相近的接受量增益，CSGB 相对固定16节点的接受量差异区间跨零；
注意力特征的独立贡献和优于所有固定预算尚未证明。速度继续门槛没有通过。
整请求 TPS 与旧 Qwen2 的 decode TPS、数据和硬件口径不同，不能跨表直接比较。

4K 注册监测矩阵已完成 336/336，请求含校准、训练、预热总计 556；
报告中的 PARTIAL 指完整 4K+8K 研究尚未完成，不能误写成当前请求记录缺失。
8K、独立确认集以及“固定 V-only mapper＋动态最大深度”的联合方案不在这次范围内。
硬件为服务器报告的 48GB RTX 4090；Target eager FP32、Draft FP16，TF32 关闭。

### 本地同步后的核验与来源

已运行归档自带的 CPU 核验，校验 87 个归档文件、59 个保留的 stage 文件、
52 个执行源码的绑定，以及 585 个原始 JSON；保存的 556 个请求均匹配对应 AR 参考。
使用 NumPy 重算了 10,000 次配对文档 bootstrap，主要点估计和区间与报告一致。
这是保存证据的哈希、统计和输出记录核验，不是重跑 GPU、验证基础模型文件，
也不是重建完整 GPU 注意力或训练梯度。归档内文件保持原始字节，不修改冻结报告。

入口：[完整报告](../exp/specextend_bridgekv/csgb_acceptance_pilot_v3/analysis/REPORT.md)、
[统计结果](../exp/specextend_bridgekv/csgb_acceptance_pilot_v3/analysis/summary.json)、
[归档核验脚本](../exp/specextend_bridgekv/csgb_acceptance_pilot_v3/verify_archive.py)。

下面第 1–8 节仍用于记录原来的三模块论文主线和 Qwen2 机制实验。
本次同步没有启动新 GPU 实验，也没有把 CSGB 当成完整论文方法成功的证据。

随后按用户请求完成了[三模块调度核查](SCHEDULING_AUDIT.md)：对齐目标执行顺序、
检查 Qwen 联合路径和 Vicuna 预算路径，并从保存计时重算瓶颈。该核查不构成新的
模块有效性或优化后加速结果；CPU 重算入口在 [scheduling_audit](../exp/scheduling_audit/README.md)。

用户随后要求先优化执行顺序与 pipeline，已实现
[首 token 重叠执行原型](../exp/attention_pipeline/README.md)，
并完成[小规模执行对照](../exp/attention_pipeline/REPORT.md)。
这是 Qwen 训练文档1500/1501上的48次计时请求，全部匹配 Target；新执行模式的
长度/接受轨迹一致。缓存注入的串行版本在0ms条件下最快；异步相对串行缓存
在0ms下decode TPS为−2.99%，在模拟每轮10ms分发延迟下为+7.56%
（含prefill总TPS分别−2.09%/+5.32%）。10ms是人工延迟，无真实边云网络证据。
原新循环与历史实现的cost输入/capture/同步不同，不能把全部变化归因于流水线。
新选择准则、注入后长度预测、独立测试和对Target AR的净加速仍未验证。

用户进一步要求完整系统实现，当前执行入口已扩为 [rescue](../rescue/README.md)，
架构在 [SYSTEM_DESIGN](SYSTEM_DESIGN.md)，结果在
[系统验证报告](../exp/system_validation/REPORT.md)。这部分为当前工作区的新代码，
不应把它误当成远端`9740a04`已经包含的内容。
系统包括真实token/chunk选择、注意力加权内容覆盖、V-only门控、预分配原生KV、
每轮工作KV、已提交位置映射memo、并行规划、长度TCP服务、回退与完整验证/回滚。

现有验证文档1554–1561，8篇、两次计时重复、64输出上限，五模式共80次请求全部
匹配独立Target参考。兼容模式保留旧选择/成本/eager：16/16组长度/提出/接受轨迹相同，
相对上一版串行缓存pipeline的文档配对平均decode TPS提高14.90%[14.48,15.52]，
整请求TPS提高10.69%[9.29,11.72]。完整coverage配置同时改变刷新和成本，配对提升
分别28.93%[25.83,32.65]/19.79%[16.88,22.52]，不能全部归因于单一模块。
池化TPS为prototype23.995/18.618，coverage30.652/22.381，Target47.465/31.053
（分别decode/整请求）；配对均值与池化比值不能混用。成本只在训练1500/1501校准。
18项CPU检查及独立保存记录核验通过。独立CPU进程完成330次真实loopback TCP长度推理，
零错误回退，且与本地coverage的16/16组轨迹一致；没有完整边云模型/KV分区。
这些是复用验证集的系统证据，尚无新独立确认集或注入后重训长度模块的结果。

随后按用户要求补充[原生SD与SpecExtend检索的匹配对照](../exp/system_validation/BASELINE_COMPARISON.md)。
同一Qwen2/H100/精度、8篇验证文档、两次重复、64输出上限；固定长度只在训练
1500/1501比较1/2/4/8/12后锁定，原生和两种检索均选中4。原生关闭全部干预与
专用观测，直接读取完整原生KV。SpecExtend仅移植chunk32、四轮刷新、Target最后
query注意力检索到同一线性骨架，Top16/32分别为最多512/1024历史token；没有复现
原版Vicuna树解码、Flash Prefill与Hybrid Tree Attention。

本轮池化decode/整请求TPS：原生34.818/24.436，Top16检索34.147/24.034，
Top32检索34.085/24.043，完整RESCUE29.913/21.900，Target AR47.362/30.793。
完整方案相对原生的文档配对整请求TPS为−10.23%[-14.61,-5.55]，相对两种检索为
−8.43%[-12.92,-3.49]/−8.63%[-11.41,-6.04]。接受率虽为65.64%而原生为53.98%，
其验证轮数为330而原生258；长度不同，不能把接受率差直接解释为草稿质量收益。
完整方案与上一轮coverage的16/16组长度/提出/接受轨迹相同。本轮未做各模块固定
长度成本消融，不能把全部损失归因于单一模块。21项CPU检查通过，60训练侧及96
验证侧计时请求全部匹配Target，27份冻结源码与完整记录核验通过。
这明确限制了前面的系统内正收益：**完整方案目前尚未超过原生SD、同骨架检索
或Target AR；没有相对原版完整SpecExtend的直接结论，也没有长上下文加速证据。**

用户进一步要求核查关键路径并优化，已完成
[第二轮系统优化](../exp/system_validation/SYSTEM_OPTIMIZATION.md)：直接GQA、原地
工作KV gather、静态/动态门控仿射分解、延后读取计算、跳过空门控及必要缓存补算
中的无用词表计算。公共修改同样应用到原生/检索基线。保留`optimized_execution`
及`--legacy-execution`，默认开启新执行；没有悄悄替换风险权重或调参结果。

约1–2K的相同8篇验证、两次计时重复、64输出上限，八条件共128/128输出一致。
训练前4篇1500/1501/1503/1504用于风险/成本校准，后4篇1505–1508用于固定长度
选择，全部与验证不重叠；四个固定家族都选中4。纯执行与legacy的16/16组长度/
提出/接受轨迹相同，配对decode+4.94%[4.81,5.09]，整请求+3.38%[2.81,3.82]。
新批次池化decode/总TPS：legacy30.608/22.285，只改执行32.105/23.066，
固定4消融32.288/23.113，校准动态长度30.609/22.256；同等优化原生SD37.356/
25.654，检索Top16为36.669/25.281，Top32为36.600/25.298，Target47.114/30.675。
固定4仍260轮、原生258轮；当前选择+注入组合没有用更少验证轮数回收额外成本。
风险校准使330轮变266轮，但Draft forward从890增至1140，未显示稳定额外加速。

25项CPU检查及29份冻结源码核验通过；128验证、92训练选参、4风险采集共224个
保存输出全部一致，风险校准在CPU重算一致，208个投机请求UUID唯一。
该结果支持执行改进，不支持超过原生SD、完整SpecExtend或Target，也不证明某个
单独模块无效。固定长度是消融；两参数风险校准仅28个训练轮次，不是新预测器
泛化成功。原始证据与限制详见上述报告，不将跨批次绝对TPS变化归因为优化贡献。

用户随后要求进一步分析权重与组件组织，新增[吞吐实现分析](THROUGHPUT_ANALYSIS.md)
及[CPU重算](../exp/system_validation/throughput_forensics_20261017.json)。固定4对原生的
3.327秒decode差额中，代数分解约95%来自更高平均轮成本，不是单模块因果归因。
以实际动作/补算回算的请求成本MAPE约0.49%–1.94%，不能把主要问题直接归于总
成本拟合；该回算也不验证风险预测/未选动作。代码确认mapper训练在Teacher选择
与原生Teacher轨迹状态上，而当前是联合覆盖与注入滚动状态；M2只按Teacher注意力
再选位置，M3不直接输入当前注入结果。候选包括按需重建唯一已提交注意力探针、
静态门控仿射memo、增量工作KV和真实计算路由。**这些尚未实现/测量**；本次只有
CPU分析，没有新增GPU或训练结果，也不自动授权下一轮实验。

用户随后明确要求按分析优化，已完成
[执行复用与注入路由实验](../exp/system_validation/EXECUTION_REUSE.md)：默认按需
重建下一轮唯一完整键探针、已提交位置静态门控仿射memo、两次跨层packed KV读取；
增量slot缓存作为可选项，重绑定前恢复旧注入V，拒绝尾部与新请求清理有独立测试。
上段“尚未实现”描述的是该CPU分析发布时的状态，不是当前实现状态。

训练1500/1501选择packed而非incremental，后者虽少读76.74%位置层但未更快；
前8篇训练文档比较9种固定路由，非零最优为零基层27、64位置、强度0.5、Teacher
排名，允许零注入时则选择关闭。联合排名可共享M1的T/S注意力评分，但不是完整
覆盖边际增益，也未训练新的动态计算路由器。网络权重与风险cost保持冻结。

现有验证1554–1561八篇、两次重复、上限64，12条件192/192匹配Target；原风险
与固定4各16/16组长度/提出/接受轨迹与本轮旧执行对照相同。原风险纯执行的文档
配对decode/整请求TPS提高1.15%[1.08,1.21]/0.99%[0.72,1.24]；固定4分别提高
0.92%[0.65,1.11]/0.67%[0.49,0.82]。完整键探针分别890→314、1102→244，
静态内容仿射位置层168,960→33,312、133,120→31,136，调用下降不等于同幅度加速。

该批池化decode/整请求TPS：新原风险31.453/22.727，新固定4四层31.415/22.721，
最后一层64位置32.786/23.408，coverage无注入33.075/23.536；同等后端原生SD
35.948/25.003，检索Top16为35.640/24.843、Top32为35.588/24.799，Target AR
47.015/30.647。轻量注入相对新四层固定4的文档配对收益为4.36%/2.91%，但整请求
相对原生为−6.15%[−7.65,−4.74]、Top16检索为−5.13%[−8.97,−1.17]；与无注入
相比为−0.66%[−1.40,0.12]，没有确立保留非零注入的净加速。

同选择固定长度的M2开关消融中，coverage四层注入接受率52.05%→54.29%，但轮数
只262→260、整请求慢3.46%；Teacher选择下四层注入也慢5.04%。它们是当前配置
的净系统效应，不是纯mapper算子耗时，更不否定旧协议的接受正信号。本轮没有
完成联合rollout mapper训练或注入后M3重训，不能把执行优化写成已解决训练适配。
31项CPU测试和32份冻结源码/284个保存输出核验通过，268个投机UUID唯一。
默认仅开启执行复用，保留四层注入与旧权重；轻量路由是显式实验选项，没有自动
删除模块。仍无长上下文、独立确认或完整SpecExtend超越结论。

用户随后授权按顺序做长文本整体优化，并明确要求**先只与SpecExtend比较**，
达到明显收益后再扩展其它性能基线。新增[长文本实现与协议](../exp/long_system/README.md)
及[当前报告](../exp/long_system/REPORT.md)。高效注意力、紧凑Q快照、分层近似覆盖、
相对覆盖/漂移刷新、末层真实SD状态重放、首token注入条件36维长度模型和共同
投影打包已实现；不是仅有计划。分层覆盖改变了原覆盖目标，需要单独解释。

新GovReport自然长报告按原train/development/test划分12/6/12篇，文档SHA256互斥；
输入2K/4K/8K/16K，未重复文本。第二轮6篇训练、4K/16K、256输出、24次采集
全部一致；mapper监测损失0.920708→0.911360，风险模型1811个轮次、监测NLL
0.603762。这是训练内代理目标改善，**不是接受率/系统加速证明**；长度采集状态
尚不是更新mapper后的新一轮交替rollout。第一轮mapper未通过监测，负结果保留。

比较包括Qwen检索移植与完整树算法流Qwen适配，后者保留树/检索/Hybrid路径，
修改QKV bias、RoPE、GQA缓存/精度、词表，使用共同Target prefill与FP32 efficient
SDPA前缀/LSE，Draft保留窗口512 Flash prefill。它不是原Vicuna/FP16复现。Target
权重/decode FP32、前缀prefill BF16，Draft BF16、TF32关闭；不能与旧全FP32
实验绝对TPS直接混算。首轮全BF16有输出分歧，仅保留失败诊断，不称无损加速。

初轮开发矩阵`exp/long_system/out_comparison_v2_20261017`已完成114/114一致请求，
独立test计时被延后，记录在`DEVELOPMENT_STAGE_COMPLETE.json`。补充开发批次
`out_comparison_v4_20261017`加入50/64树节点、固定2、相同1024预算无注入等控制，
以各批次同时测量的检索锚点归一化后选配置，74份执行源码冻结。三篇开发
8K/16K选配置后冻结八篇独立test、四长度、两次重复、256输出，门槛为8K/16K
相对两种择优SpecExtend均至少+10%且配对区间下界>0。40项CPU检查和训练核验
通过。**正式矩阵尚未完成，不写成已经超过SpecExtend/SOTA。**
首版因重复打包多占18.7GB而在开发/测试计时前中止；共享后重建第二版，原数据
与旧归档不改写。未运行EAGLE-3、DFlash等其它性能基线，符合用户当前范围。

2026-10-18用户要求先暂停查看进展，测试GPU进程和自动核验/绘图队列均已停止，
不自动恢复。初轮114＋补充84个开发请求完成；独立测试71/320，已保存71个输出
全部匹配Target，完整八文档统计尚未生成。暂停元数据见
[PAUSED.json](../exp/long_system/out_comparison_v4_20261017/PAUSED.json)。开发选择冻结：
非零为1024预算、Draft注意力权重0.25、固定4；零允许为相同预算、权重0.5无注入；
动态单独保留512预算。开发整请求TPS分别24.508/24.781/22.107，较快检索为24.835，
完整树为20.479。它们不是独立测试结论，非零/零允许还混有注意力权重差异。
当前没有达到明显超过SpecExtend的目标，不能只与较慢树移植比较后宣称成功。
原脚本拒绝已有输出目录，恢复时须保存已完成记录并明确接续协议。

用户随后明确要求把调度与模块目的分开评价/优化，已完成新的
[模块目标框架](MODULE_OBJECTIVES.md)及[三篇新开发文档pilot](../exp/module_objectives/REPORT.md)。
这不恢复原暂停测试；新开发229/231/251、8K/16K、96输出、两次计时，132/132
SD输出匹配Target。共同Teacher前缀/原生Student历史下，144个开发query的完整
读取/分层512为121/125次贪心命中；旧/新mapper均125，没有纠错/伤害，只有KL
下降。局部诊断不覆盖真实滚动历史及四轮刷新，滚动系统仍慢于完整原生读取。
原生完整backing KV与prefill保留，不能把512工作位置说成物理缓存缩为512。

新margin＋原生正确间隔保持训练的M2在48个训练监测query仍为37命中；放大强度
也未增加命中。Target已知、同层同位置有界V直接搜索达到40/48，只证明其中3个
状态存在纠错方案，不是学习到的结果、全局上界或Teacher V内容兼容性证明。
M3同配置重算表明旧动态能少浪费但多验证；新效率目标标定相对旧长度配对TPS
仅+0.17%且区间跨零，仍比固定4慢约2.30%。没有独立注意力特征归因。

已实现首token末层attention内提前发布注入摘要、host/stream阶段计时、工作量
指标和训练内条件hazard先验混合。早发布相对异步晚发布+0.18%，仍比串行慢1.83%，
12/12组轨迹一致；默认auto继续使用单卡串行。36维模型仍只支持本地调度。
45项CPU测试、51份冻结源及132输出/1680诊断记录核验通过。冻结训练源只把
基模型排除在优化器外，后续入口额外关闭无用基模型梯度；不混作本轮推理收益。
新协议/汇总/原始数据均在上述报告链接内，只有三篇开发，不能宣称完整方法/SOTA成功。

### 2026-10-18：按模块性能指标继续优化

用户随后明确要求“针对模块性能指标，进行优化”，已完成
[优化实现与结果](../exp/module_opt/REPORT.md)。这是独立于暂停测试的新开发研究，
原71/320不恢复；无其它外部baseline扩展。本次确认303/306/327取自原GovReport
validation的自然长报告，完整文档SHA及8K/16K token前缀与long_system旧三划分
不重叠；训练拟合/监测/选择沿用原train的不同文档。输入8K/16K、输出128，
精度仍FP32 Target decode、BF16 prefix/Draft、TF32关闭。不得与上一轮229/231/251、
96输出或更早短文本的绝对TPS直接混算。

M1实现高联合注意力锚点、覆盖目标非负注意力质量项和更小预算，但训练侧验证
轮数253增至280–296，未通过质量门槛，因此保留原分层512；确认的`m1_old`与
`m1_new`实际同配置，不把计时噪声当作M1改善。两项新参数默认0。

M2新增32维低秩query条件向量头，使V修正方向也随query变化，保留静态仿射
缓存及原生backing K/V；末层训练用新采集的576条滚动query、匹配系统支持行64。
仅480拟合query生成Target已知的有界V辅助标签，96监测不做oracle优化。监测
仍71/96，按严格命中门槛未替换下游mapper，但预定候选保留确认。144个共同
query无注入/旧mapper/query头为105/107/109命中；query头相对无注入改对4、
改错0。实际滚动废弃930→910、验证500→496，整请求相对旧mapper+0.31%
[-0.49,+1.86]，尚无稳定速度收益，相对无注入仍慢1.60%。这是小质量信号，
不等于最终任务正确率提升，也未证明Teacher内容、精确位置或gate独立贡献。

在打开主确认输出前另冻结读取条件补充方案：增加Student实际Q/working K的
每KV-head读取权重，联合attention输出空间辅助蒸馏。没有注入Teacher K或训练
独立逐head硬路由。新确认命中104/144，相对旧mapper改对1/改错4，TPS约慢0.94%，
未进一步改善；输入、辅助目标和重复样本同时改变，不能据此否定Q/K的一般价值。

M3在实际采用的旧mapper下重采集381轮，训练排除A·V误差的34输入风险器，加入
请求内已验证条件反馈，首拒绝/EOS之后删失、新请求重置。训练选择固定4，
surplus注意力候选却在确认过长：废弃930→1232、验证500→478、TPS慢5.27%。
按训练数据预定的另一Pareto端点ratio在补充确认中废弃930→876（−5.81%）、
Draft forward2140→2096，但验证500→514，TPS相对固定4−1.64%[-3.36,+0.97]，
相对旧动态−0.12%[-2.73,+3.92]。少浪费目标有小幅进展，净加速仍未达到；
8K相对旧动态的废弃甚至上升，16K才减少。没有完成新query mapper＋新M3联合
成功；固定12采集与动态动作/刷新/历史分布之间的迁移仍待解决。

两个阶段使用同三篇确认文档、各两次随机顺序计时，不是独立复制。主运行194个
保存SD输出、补充84个全一致，其中确认180；52项CPU检查通过，五个共同条件
各12/12组长度/提出/接受轨迹跨源码阶段一致。不同阶段用各自冻结源码，后续
CLI/RPC检查及分析导出不算作已测速度优化；36槽模型及纯历史包装拒绝旧31维
TCP协议。旧默认权重不替换，风险/mapper候选需显式加载。保存记录核验、来源
哈希及限制见[最终核验](../exp/module_opt/final_audit.json)；仍无完整方法/SOTA结论。

### 正在进行：M2/M3内部架构与训练

用户继续授权优化两个模块内部设计，代码与预定方案在
[module_arch](../exp/module_arch/README.md)，当前状态在
[报告](../exp/module_arch/REPORT.md)。新增native-residual紧凑V映射、全部末层
mapper块的前缀代理训练与更新后重采集，以及生成两个token后再选择2/4长度
的12输入分类器。短轨迹保持删失，不查询未来Teacher。当前CPU检查59项通过。
确认换成604/613/614、8K/16K、128输出，原71/320仍暂停。

主运行训练中的compact保留零增量，因此即使更快也不证明有效内容补充。补充
开放门控/整段平衡采样的方案已在主确认前固定；训练最快长度候选与受约束候选
另作匹配确认。它们复用同三篇文档，不是独立复制，补充mapper不与新长度联合。
运行尚未完成时只交接实现与协议，不能据训练损失或部分输出宣称系统成功。

主运行随后完成72/72确认，180个保存SD输出全部一致、864条共同前缀和69份
冻结源码核验通过。全末层M2相对旧query头命中123/144→124/144，整请求约
+0.28%，但相对无注入仍慢1.64%。compact仍零修正。相同compact下的两token后
长度策略使废弃1054→866、验证524→526、Draft forward2236→2062，文档配对
整请求TPS+2.40%[+1.50,+3.11]，相对纯先验+3.15%。这是新三篇文档上的有限
M3正信号；含置信/注意力等联合输入、mapper零修正，不能写成注意力独立归因、
有效非零mapper联合成功或超越SpecExtend。开放门控及最快长度补充仍在运行，
更新以报告和各阶段COMPLETE/audit为准，不提前写出结论。

## 1. 用户最新确定的论文主线

研究对象是 speculative decoding（SD）中的目标模型 T 与草稿模型 S。
用户希望围绕**跨模型注意力**讲一个统一故事，而不是拼接几个独立加速技巧。

核心研究假设：

> 大小模型的输出分歧，部分可能来自对任务相关上下文的读取与利用差异。
> 利用跨模型注意力，可以尝试定位重要信息、补充 student 所缺失的内容，
> 并估计这种指导还能支持多长的可靠草稿。

对应用户的画作比喻：先帮助学生确定焦点，再补充焦点内的画面细节，最后判断
这份指导能支持他继续看/描述多久。论文正式表述使用读取、内容补充与指导有效期。

用户偏好 token 级干预。Chunk 是可调粒度，chunk 长度为 1 时就是 token；
不同粒度要比较相同实际 token/字节预算，不能只比较相同 chunk 数量。
Mapper 是内容传递的实现途径，注意力才是文章的组织主线。

注意：“性能差距一部分来自性能”是讨论中的口误，应使用上面的读取与利用差异假设。
当前贪心实验中，接受直接由连续草稿 token 是否等于 teacher argmax 决定。
注意力是拟研究的内部机制/预测信号，不能直接称为接受率的第一性原理，
更不能把注意力相似性或相关性当成已证明的因果解释。

## 2. 三个论文模块与执行关系

| 论文模块 | 核心问题 | 输入 | 输出 |
|---|---|---|---|
| 1：位置选择 | 关注哪里？ | 同一已完成 query 的 T/S 注意力、候选内容表示、预算 | 工作位置集合 R |
| 2：选择性内容补充 | 补什么、补多少？ | R、teacher 缓存、student 当前状态、层间配对 | 内容修正及 token/层/query 条件门控 |
| 3：长度控制 | 当前指导能走多远？ | 注意力覆盖/差异、参考年龄、R、补充后的 student 观测、成本 | 暂定/最终长度 k，或提前停止决定 |

用户提出模块 1 和 3 先规划，再把位置和长度信息交给模块 2。
可保留这个框架，但长度应先视为暂定预算：注入会改变 student 状态和接受情况。
一种合理的执行版本是：

1. 从 prefill 或上一轮完整验证复用 teacher 信号，获取因果可用的 student 信号。
2. 模块 1 选择 R，模块 3 给出暂定长度 k0。
3. 模块 2 按层和 token 补充内容；已生成的 student 状态可进一步更新门控。
4. 模块 3 利用补充后已经观测到的状态确认长度或提前停止。
5. Teacher 用完整原生上下文验证，提交接受前缀与 correction/bonus，回滚拒绝尾部。
6. 更新下一轮参考信号。

“注入后再确认长度”是下一版设计建议，尚未完成对应的新训练与有效性实验。
不能在未调用 teacher 时声称拿到了当前未验证 query 的真实 teacher 注意力。
历史 teacher 参考应标明年龄，额外探针、mapper、规划成本都要进入实际耗时。

执行优化的第一版保留旧预测器对R的依赖：模块1给出R后，模块3异步计算，
模块2准备和第一个正常草稿token先执行，在生成第二个token前取得长度结果。
这不是使用注入后特征重新预测，而是将原决策移出部分关键路径。每轮先重绑定
静态映射缓存，query门控仍动态计算。该首轮原型只有本地线程/CUDA流；
后续完整系统新增真实TCP长度服务，具体部署边界见第0节与系统报告。
未来边云应分别发送位置/内容与长度消息，并绑定请求、轮次、前缀和工作集版本；
否则等待打包长度会重新阻塞内容分发。

## 3. 模块一：两种选择目标必须分开

### 已完成：高注意力候选上的局部读取保持

实现：[selection.py](../exp/attention_horizon/selection.py)。
使用最后一层、逐 query head 的注意力与正确 GQA value 分组。
先以 teacher 高注意力/历史集合初始化；预算 512，固定首位和最近 64 位。
替换候选来自 T/S 高注意力位置。最多四次局部交换，目标是：

\[
\min_R D_T(R)+\lambda D_S(R),\qquad
D_M(R)=\operatorname{mean}_h
\frac{\|\sum_{i\in R}\frac{a^M_{h,i}}{\sum_{j\in R}a^M_{h,j}}V^M_{g(h),i}
-\sum_i a^M_{h,i}V^M_{g(h),i}\|^2}{\|\sum_i a^M_{h,i}V^M_{g(h),i}\|^2+\epsilon}.
\]

代码使用下界截断保护分母；公式表达其归一化读出误差的含义。
这保持的是各模型原有的 A·V 读出，不能直接解释为学生缺失的信息，
也没有显式优化内容去重。用户指出这与“注意力＋信息增量”的原意存在区别；
已承认这一点，不能在论文中混用名称或把该结果当成新方案的验证。

### 已集成、独立质量待验证：注意力加权的内容覆盖

目的：注意力表达重要性，边际覆盖表达一个 token 对当前集合增加了多少
尚未覆盖的内容。候选定义为每个模型内部的内容相似度：

\[
s_M(i,j)=\max(0,\cos(V_i^M,V_j^M)),\qquad
c_i^M(R)=\max_{r\in R}s_M(i,r),
\]
\[
F_M(R)=\sum_i a_i^M c_i^M(R),\qquad
\Delta_M(j\mid R)=\sum_i a_i^M\max(s_M(i,j)-c_i^M(R),0),
\]
\[
\Delta(j\mid R)=\Delta_T(j\mid R)+\lambda\Delta_S(j\mid R).
\]

从强制位置开始，反复选择当前边际收益最大的候选并更新覆盖，直到预算用完。
T/S 内容相似度在各自空间计算；相加的是标量收益，不直接比较不同维度的 V。
“V 相似代表内容重复”是待验证代理，不是真实语义等价、互信息或信息论密度。
表示如何聚合头、哪些历史位置参与覆盖求和、候选池大小等要在正式协议固定。

原 [attention_coverage](../exp/attention_coverage/README.md) 仍只有合成成本原型。
后续完整 [rescue系统](../rescue/README.md) 已集成该类准则，并有第0节所列的真实模型
系统测量；默认按头平均注意力、展平原生KV-head的V、远历史权重归一化、1024候选和
每批16位置更新。该结果还同时改变刷新与成本，不是覆盖准则的独立质量验证，
也不能借旧读取保持结果证明覆盖收益。
H100、1,024 候选、65 个固定位置、补选 447 个位置：

| 上下文 | 单个贪心 | 每批 16 个 | 每批 32 个 |
|---|---:|---:|---:|
| 2K | 52.3 ms | 4.1 ms | 2.2 ms |
| 16K | 298.2 ms | 20.5 ms | 10.9 ms |

包含相似度准备与选择，不含注意力采集/模型运行/缓存应用。
批量选择不重算同批内部冗余，是未验证质量的近似。复杂度：相似度 O(NKD)，
逐个贪心 O(BNK)，相似度存储每个模型 O(NK)。16K 行只是合成微基准，
不代表已做 16K 模型实验。

讨论过的后续路线：先离线验证完整准则；线上尝试高注意力核心＋64–128 个
补充位置、小批量更新，以及复用若干轮后触发重选。这些都没有新效果结论。

## 4. 模块二：层间来源、mapper 与门控各司其职

已有试验：[selective_kv/README.md](../exp/selective_kv/README.md)。
当前具体实现是 V-only：在 student 最后四层、128 个高 teacher 注意力的远历史
位置进行工作读取修正，K 不变，两基础模型冻结。

层编号使用人类的 1-based 编号：

| Student 层 | 25 | 26 | 27 | 28 |
|---|---:|---:|---:|---:|
| Teacher 来源层 | 25 | 27 | 27 | 28 |

当前映射方向是 **student 层 → teacher 来源层**，多个 student 层可以共用同一
teacher 层。它没有验证“把多个 teacher 层融合到一个 student 层”的方案。
层间相似性提供候选来源先验，不证明内容天然兼容；两模型本身均为 28 层，
不能宣称已验证不同层数模型的泛化。

每个注入层的 mapper：LN(512) → Linear(64) → SiLU → Linear(256)，
将 teacher 的 GQA value heads 展平后映射到 student 维度。
Mapper 决定内容修正方向，门控决定应用强度：

\[
V^{\ell}_{\mathrm{read},i}
=V^{\ell}_{S,i}+g_{\ell,i,t}\,\Delta V^{\ell}_{i}.
\]

修正经过 native V 的 RMS 定标及有界 tanh。门控使用 teacher latent、native V、
当前 student 深层 query 状态、位置/注意力及相似度；是 token/层/query 条件的
连续强度，不是硬选层，也没有独立的逐 attention-head 门控。
当前可注入层段与来源配对是固定的，未训练一个自动搜索全部层的选择器。

训练目标：KL(T||S)＋0.25 teacher-argmax CE＋0.01 残差惩罚。
这是可微输出性能代理，实际 greedy 接受率另外测量；不是直接对离散接受率求导。
校正只作用于工作读取，不覆盖 backing KV；后续生成的原生 student 状态仍可能
受前面注入的内容影响。更一般的 K+V 内容注入尚未在此 pilot 中验证。

## 5. 模块三：已有预测器及其限制

实现：[risk.py](../exp/attention_horizon/risk.py)、
[runtime.py](../exp/attention_horizon/runtime.py)。
分别比较结构特征、置信度、注意力、注意力＋读取误差、全部特征。
两层 32 维 MLP 为未来 12 个位置预测“继续接受 / 接受 EOS / 拒绝”三类条件风险。
未观察尾部右删失。每组训练三个种子，验证选择 checkpoint、温度和种子。

用训练测得的成本，从长度菜单 {1,2,4,8,12} 选择预计提交效率最高的 k：

\[
\frac{1+\sum_{j=1}^{\min(k,r-1)}P(\text{前 }j\text{ 个均接受且非 EOS})}
{\widehat C(R,k)}.
\]

r 为剩余输出预算；求和上限避免错误计入预算末尾 bonus。
已测边界决策及额外的逐 token 更新版本，历史 teacher 参考不被当成当前真值。
新架构中对“内容注入后状态”的专项训练/校准仍待做，不能用旧长度结果代替。

低开销强对照包含最佳固定 k、历史验证风险 EMA、关闭不需要探针的置信度预测器。
用户希望先把长度模块做好，再评估联合决策；这是后续优先方向，尚不是成功结论。
联合与顺序都用同一预测器时，联合额外成本仍是独立问题，不能全部归因于长度模块。

## 6. 证据清单：写作时可以说到哪里

**论文模块 1/2/3 与实验 A/B/C 不是同一套编号。**
实验 A/B/C = 读取保持选择 / 长度 / 两者联合；V 注入属于独立 pilot，
在 C 中只补做了冻结 mapper 集成。

| 证据 | 已有结果 | 不能推出的结论 |
|---|---|---|
| Token vs chunk，阶段二 B=512 | 接受率 59.62→62.22%；decode TPS +3.17%，CI [+0.57%,+5.99%] | Token 所有预算都更好、或优于原版 SpecExtend |
| 峰值并集 | 阶段二相对 teacher-token 没有稳定额外收益 | 联合注意力已稳定优于 teacher 单独筛选 |
| 层间关系 | 测试 JS：相似配对 0.173，深度配对 0.208，随机 0.285；最近配对一致率约 36% | 层间语义/内容等价、映射已稳定泛化 |
| 性能监督 V 门控 pilot | 63.51→65.26%，+1.75pp，CI [+0.53,+3.52] | 任意 KV 注入有效，或门控本身的独立收益已证明 |
| V 来源诊断 | 打乱内容 65.39%、均值内容 65.44%；对齐 65.26% | 精确 token 级 teacher 内容对应已证明 |
| 新读取保持选择 A | 57.91→58.60%，+0.69pp，CI [-1.08,+2.66]；吞吐 -5.08% | 新信息增量覆盖算法有正收益 |
| 新动态长度 B | 验证 565→384 次；同探针相对最佳静态 TPS -0.67%，CI [-4.06%,+2.57%] | 少验证等于加速 |
| 动态长度实际成本对照 | 相对低开销最佳固定长度 TPS -8.69%，CI [-10.16%,-7.32%] | 只需调参即可保证成功 |
| 联合决策 C | 相对顺序 TPS -9.14%，CI [-11.21%,-7.08%]；交互项 CI 跨零 | 已证明联合优化有额外收益 |

V pilot 的筛选基线/门控 decode TPS 为 29.74/27.20，尚未加速；同 mapper
恒定门控及打乱/均值/零来源的解释见原报告，不能挑选一个正向消融概括全部归因。
新 A/B/C 中 teacher decode TPS 46.90，最快 SD 对照 32.63，未超过 teacher。

上述阶段二、V pilot 与 A/B/C 的范围：单 Qwen2 模型对、H100、约 600–2048 输入；
teacher FP32 / student BF16。新增 Vicuna 4K CSGB 归档的范围另见第 0 节。
阶段二和 V pilot 最大输出 32，新 A/B/C 为 64；数据划分也不同，不能跨表直接作
因果比较。新 A/B/C 使用全新 48/16/30 文档，标题互斥且排除旧文档。
720/720 次该轮主测试 SD 匹配 FP32 teacher 参考；这证明本轮经验一致性，
不保证其他精度、采样方案和实现天然一致。原 BF16 数值分歧另有审计，不要隐去。
区间未作多重比较校正；预测器三个种子不等于整个系统或 mapper 的三次独立复现。
完整 backing KV 和 student prefill 仍保留，不能称为物理缓存压缩或移除 prefill。

## 7. 新会话从哪些文件取证

| 用途 | 入口 |
|---|---|
| 整体研究历史 | [实验进度.md](../实验进度.md)，尤其第 8–10 节 |
| 筛选粒度、并集、层间对应 | [阶段二结论](../exp/attention_selection/out_stage2/conclusions.md)、[完整报告](../exp/attention_selection/out_stage2/report.md) |
| Mapper、门控、来源消融 | [V pilot 结论](../exp/selective_kv/out_pilot/conclusions.md)、[来源报告](../exp/selective_kv/out_pilot/source_controls_report.md) |
| 已完成的 A/B/C | [实验说明](../exp/attention_horizon/README.md)、[结论](../exp/attention_horizon/out/conclusions.md)、[报告](../exp/attention_horizon/out/report.md) |
| A/B/C 数值及核验 | [summary.json](../exp/attention_horizon/out/summary.json)、[completeness.json](../exp/attention_horizon/out/completeness.json) |
| 新覆盖准则仅有的成本测试 | [说明](../exp/attention_coverage/README.md)、[合成结果](../exp/attention_coverage/out_benchmark/summary.json) |
| 完整三模块执行系统 | [实现](../rescue/README.md)、[设计](SYSTEM_DESIGN.md)、[系统报告](../exp/system_validation/REPORT.md) |
| 新增 Vicuna 4K CSGB 内部预实验 | [归档入口](../exp/specextend_bridgekv/csgb_acceptance_pilot_v3/README.md)、[报告](../exp/specextend_bridgekv/csgb_acceptance_pilot_v3/analysis/REPORT.md)、[核验脚本](../exp/specextend_bridgekv/csgb_acceptance_pilot_v3/verify_archive.py) |
| 历史 SpecExtend/BridgeKV 路线 | [入口](../exp/specextend_bridgekv/README.md)，不能与新 Qwen2 路线混算 |

`exp/attention_horizon/out/mechanisms.png/.pdf` 是已完成实验的结果图，
不是整篇新架构已经有效的证据。原始逐题 JSONL 在对应 out 目录。
历史报告和执行时哈希保持历史含义；后续源码更新不应伪装为过去运行的代码。

## 8. 撰写论文的建议起点

先写核心问题、假设和方法框架，再写有证据支撑的动机实验。
Introduction 可围绕注意力指导的信息选择、内容补充和有效期控制提出统一研究问题；
Method 明确定义 R、层间来源、内容修正、门控、长度及推理时可用信息；
Experiments 按对应研究问题排列，而不是按开发时间排列。

“高注意力覆盖的信息是否冗余”“层间关系能否指导可用内容传递”
“指导失效能否被注意力预测”“联合决策是否值得成本”是待检验的问题。
贡献先写为候选方法/研究设计，不预填 SOTA、加速比或因果结论。
实验结果未完成的段落用明确待验证标记，不以愿景代替实测。

待补实验证据的重点：

1. 新覆盖选择与同预算 Top-k/并集、打乱内容相似度的比较；先看质量再看成本。
2. 公平训练的 native-only V adapter / 文档整体条件化，分离一般适配与精确内容传递。
3. 层对应与注入兼容性的独立消融，明确多对一的方向。
4. 离线最优长度诊断、低成本注意力特征、固定长度回退及注入后状态校准。
5. 在长度模块具有净收益后，重新检验联合决策，并单独控制候选搜索成本。
6. 最后再扩展长上下文、不同层数模型、更多系统训练种子与优化执行后端。

本文的研究清单不自动构成下一轮实验任务；后续用户授权的执行更新及范围见第0节。
新会话若需要文献定位/创新性比较，应查原始论文核对来源；本交接不提供未经核实的
相关工作断言。现有 teacher-chunk 是 Qwen2 线性 drafting 的 SpecExtend 式对照，
不是原版 Vicuna 树解码复现。

本地实验 Python：`/home/wanghl/miniconda3/envs/rescue/bin/python`；
绘图另用 `idn` 环境。模型路径为 `/mnt/8T/cl/models/Qwen2-7B-Instruct` 与
`/mnt/8T/cl/models/Qwen2-1.5B-Instruct`。权重、张量缓存和日志不在 Git 内，
从仓库取得代码不代表已经取得这些本地文件；写论文无需重新启动 GPU 实验。
