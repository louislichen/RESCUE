# FP32 split-KV：默认关闭的执行候选与真实检查状态

最新状态：`MATCHED_DEVELOPMENT_TIMING_COMPLETE_GOAL_NOT_MET`。48次质量门禁和792次公平计时均已完成、核验、出图和只读封存；最终Token全部一致，但原六长度相对同执行检索Graph基线超过10%的目标未达到。

## 2026-10-21 实际更新

用户已明确允许中间浮点末位变化，但要求 FP32、关闭全局 TF32，并保持所有评测输出 Token 与冻结原 Target 完全一致；同一执行实现必须公平提供给检索基线。旧实验自己的严格门禁不变。

在同一 H100 的三个新隔离目录分别运行了算子检查，不修改旧源码、权重、Mapper、策略、成本模型或新确认数据：

- v1：实际编译并执行 100/112 个用例，在单键用例099失败。候选逐位等于解析值，原 FSDPA 参考不等于解析值；差值最大为 `4.76837158203125e-7`。未运行剩余12例。
- v2：只将单键用例 Q/K 归零，保持 V、随机数消耗、掩码与双方解析值严格门禁不变；仍在099以相同输出与误差失败。这否定了本次 Q/K 夹具修复假设。两次 FAIL 均完整保留，未重启或重分类。
- v3：事先明确更正新检查合同，112个输入和候选算法与v2完全不变。候选对解析值仍严格逐位；原 FSDPA 对解析值仅作为诊断，随机96例仍不设数值等价通过门槛。实际112/112执行及同步完成，16个候选解析用例均通过；参考12例匹配、4个单键例不匹配。状态限定为候选解析 smoke，不是参考等价、模型 Token 或速度通过。

三次均正常结束或失败退出并已收集保存结果；未测接受率、模型输出、延迟或 TPS。实际 JIT 缓存条目 v3 最后为 stage_one12、stage_two1，条目数不是编译耗时或速度。详见 [v1失败报告](out_fp32_splitkv_operator_20261021_v1/analysis_v1/REPORT.md)、[v2失败报告](out_fp32_splitkv_operator_20261021_v2/analysis_v1/REPORT.md) 和 [v3原始终态](out_fp32_splitkv_operator_20261021_v3/results/TERMINAL.json)。

v3 的 [报告与三套图](out_fp32_splitkv_operator_20261021_v3/analysis_v1/REPORT.md) 已生成、实际查看并从保存的独立 CLI 重跑 summary 字节一致；v1/v2同样保留明确 PARTIAL 的失败图。三目录均有本地 SHA 清单与可恢复的0444只读保护，不冒称操作系统 immutable。

后续模型门禁于2026-10-20 17:04:16 UTC正常exit0，原tmux/pane242384/模型PID242390均已结束：旧doc271/303、六个实际输入长度、两种Graph方法×原实现/候选，共48次独立质量owner，全部输出逐Token等于旧封存canonical，48owner全部关闭。24对接受/工作轨迹、DraftQ及NativeS相同，NativeT浮点字节均有不同；不把中间末位变化误作新合同失败，也不宣称一般随机分布等价。实际29344次候选Target层调用、672次原算子fallback。31712项独立saved-record审计、7项报告CPU测试、最终五套PNG/SVG实际视检与保存CLI逐字节summary复现均通过。详见[最终质量报告](out_fp32_splitkv_model_20261021_v1/analysis_v2/REPORT.md)、[原始记录](out_fp32_splitkv_model_20261021_v1/results/QUALITY.jsonl)、[独立审计](out_fp32_splitkv_model_20261021_v1/audit_v1/AUDIT.json)。该轮带质量观测，TPS/延迟均未测。

模型核验根已完整本地封存，manifest SHA `013273e977a8b4aeb97b3ceff845bc9b03ffbe0ab7f4752307c65c5b598f7320`。148个取回文件的完整SHA与size在正式传输前/后及本地相同；早一次SCP未完时的本地检查正确拒绝，未重跑或覆盖实验。

新的[792请求公平计时](fp32_splitkv_timing_v1_preregistration.json)于2026-10-20 17:58:44 UTC正常exit0。实际tmux/pane334007/模型PID334017及子进程均已结束、GPU空闲；exit0绑定同一terminal。两method均比较原/候选执行，六长度×16旧文档×2重复，共24owner，每owner旧doc271冷prime+同32warm；所有晚capture仍计费，Mapper/策略/成本/文档冻结。全部792输出匹配冻结原canonical，384组原/候选工作量和语义轨迹均未变。

两边都使用新Target执行的主收益（2K/4K/6K/8K/12K/16K）为 **+6.120/+3.602/+1.977/+4.777/+1.557/−3.980%**；点态95%文档配对CI依次为[0.526,12.365]、[−2.734,11.475]、[−3.081,8.001]、[0.389,10.003]、[−2.444,6.067]、[−7.862,−0.301]%。自身执行加速8.006%–32.114%，但检索同样加速6.560%–31.828%，不能把共同加速当作相对收益。16K我方1462轮对基线1404轮；平均完整generate秒/轮近乎相同，此为账面分解，不是模块因果归因。

34份完整源码/raw共83,318,496 bytes经过实际SCP退出0、传输前/后和本地逐SHA+size相同；25891项独审与1293项独立报告数值对照通过。主线程12/12报告测试、直接raw配对CI复算、九套最终PNG实际查看、独立保存CLI summary逐字节复现均通过。保留原显示v1，v2仅修图例及坐标留白。162份最终产物0444可恢复只读，manifest SHA `9b727ca38bea86343dd6d5004342da2969ff3b6f1b4d605016445fdf967a47bd`；不是OS immutable。

[最终报告](out_fp32_splitkv_timing_20261021_v1/analysis_v2/REPORT.md) · [公平收益图](out_fp32_splitkv_timing_20261021_v1/analysis_v2/relative_tps_gain.png) · [接受长度](out_fp32_splitkv_timing_20261021_v1/analysis_v2/acceptance_length.png) · [TPS](out_fp32_splitkv_timing_20261021_v1/analysis_v2/warm_tps.png) · [分析源码](out_fp32_splitkv_timing_20261021_v1/analysis_v2/analysis_entry.py) · [raw](out_fp32_splitkv_timing_20261021_v1/results/RUNTIME.jsonl) · [终态核验](out_fp32_splitkv_timing_20261021_v1/VERIFIED_COMPLETE.json)。这是复用开发文档、非交错owner、有限cold+32warm的测量，不是新确认/完全预热生产/SOTA；总体目标仍active，暂停确认数据未动。

固定 PyTorch 源码的 [float/SM80 GEMM 分支](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h#L100-L113) 声明 `OpMultiplyAddFastF32`。因此不能从 FP32 张量和全局 TF32 flag 反推旧定制算子的所有乘法均为严格 IEEE FP32；本次未采实际指令，不将误差武断归因某条 reciprocal 或 Tensor Core 指令。

## 初始离线准备记录（早于上述 GPU 检查）

- 新增 [独立算子](fp32_splitkv_candidate_v1.py) 和 [CPU 契约测试](../../tests/test_fp32_splitkv_candidate_v1.py)。不接入 Engine，不替换任何已冻结 attention/runtime，不改 Mapper、策略、旧成本模型、权重或历史结果。
- 主线程实际执行 `python -B -m unittest tests.test_fp32_splitkv_candidate_v1 -v`：最终 16/16 通过。测试是接口、元数据、隔离和源码约束检查，**没有执行 Triton kernel**。
- 独立审查发现并修复两项 P2：拒绝带未解析负位/共轭位的指针输入，拒绝已加载的非默认 Triton cache manager；也拒绝空字符串的 manager override。补充 CPU 回归后重新运行，不把这些边界测试冒充真实模型故障。
- 数值审核的 5/5 stdlib CPU toy 测试也通过；它只否定普通 split 合并的一般逐位保证，不证明真实 Qwen Token 会改变，也不证明新算子会更快。
- 本次没有 SSH、GPU、模型调用、Triton 编译、CUDA 数值测试或端到端计时；没有打开暂停/新确认数据，没有安装依赖。

## 固定接口与范围

`build_candidate(fallback, *, enabled=False, cache_dir=None)` 在默认关闭时直接返回原 `fallback` 对象，不包装、不导入 Torch/Triton、不分配 Tensor。

显式开启后的调用只支持 folded FP32 Q/K/V：Q 为 `[4,7,Q,128]`，`Q=2..5`；K/V 为 `[4,7,N,128]`，`Q<=N<=65536`；允许广播 head 的 stride 0。可选 rank-4 FP32 additive mask 或 bool mask，`dropout_p=0`、`is_causal=False`，由传入 mask 表达原来的绝对位置因果关系，不另造 RoPE 或 cache。

候选固定 8 个分区、每块 128 **keys**，不是 128K Token。两阶段保存/合并 FP32 max、分母、加权 V；不用 `tl.dot`，不改成 BF16 或 TF32。N 和相关 stride 声明为 runtime/non-specialized 参数；**CPU 检查不能证明实际编译次数或 CUDA 数值行为**。

开启后的真实 CUDA 调用必须在新的 owned 子进程内，已有显式 owned cache，环境已精确指向该目录并关闭 driver cache；实现检查默认 Triton cache manager 状态，不创建目录、不修改环境、不静默 fallback。这里不承诺任意被 monkeypatch 进程的隔离。当前不支持 CUDA Graph capture；也没有跨调用复用 scratch 的性能证明。

## 初始动机与最终速度结论的区别

现有单文档诊断支持调查小 Query / 长 KV 的 Target 注意力并行，但活动时长不是可直接删除的墙时。若该 Target 优化适用于两边，检索基线也必须获得，不能用旧执行对照宣布相对收益。

原 16 文档、每文档两次的保存数据中，若仅按“基线完全不变、输出 Token 数不变”计算到 +10% 点估计的时间差，4K/8K/16K 分别还需约 0.119/0.131/0.667 秒每请求。**这是旧数据的条件算术，不是候选加速预测、置信区间或新实验。** 共同 Target 优化甚至可能缩小部分长度的相对优势，不能只报告它本身的 kernel 加速。

重复项已经排除：Coverage greedy-loop Graph、内容归一化缓存和 attention mean 复用均已有实现，不能再作为新成果。M1 粗选计数几何化是尚未测量的小候选，当前没有把它包装成完整 M1 的节省。

## 后续真实验证仍缺什么

1. 数值验收已由用户确认：中间末位差异允许；FP32、关闭全局 TF32、最终评测 Token 与冻结原 Target 完全一致。旧实验严格门禁不重写。
2. 上述算子 smoke 已完成；它覆盖非方形 mask、尾块、空/全 mask 分区、广播 stride与输入不变记录，但不等于一般数值等价。Tensor 值与编译二进制未取回，本地审计只核保存记录与来源。
3. 对冻结原 Target 的 canonical 输出做真实 Token 核验；披露中间数值对 M1 选位、Mapper 条件、策略动作和逐轮工作量的影响，不能以候选自己的 AR 自证。
4. 两臂使用公平执行条件后，回到原六个实际输入长度的端到端配对比较。保留独立接受指标、TPS、真实捕获/初始化费用、机制诊断和文档级置信区间；不能用 kernel 微基准、单文档诊断或普通 SD 替代。
5. 上述792轮的图、源码、报告、独审和封存已完成；它证明本次执行实现使两臂加速，但没有达到原相对>10%目标。后续新候选仍须独立门禁和公平实测，不能继承本轮性能通过。

## 代码身份与已有证据

- Candidate SHA256：`9f133cc824915afd09d9ad25a1b16e6cd721522e1f583cabd5e102235358cfd2`
- CPU tests SHA256：`d05aab7170f75451f1af4e93aac9a895c25e9b3f15ae9c59d11e2b6117bed8c0`
- [浮点归约审核](numerical_contract_review_20261020_v1/REVIEW.md)
- [原诊断与原始数据入口](out_critical_path_probe_20261020_v2/DIAGNOSTIC_FINDINGS.md)
- [原六长度结果](../context_curve/out_h100_context_curve_v2_20261020/terminal/results/analysis_final_v2/REPORT.md)

Triton 参数声明按服务器版本 3.3.1 的 [JIT 源码](https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/runtime/jit.py) 检查；这不是该服务器实际编译候选的证据。
