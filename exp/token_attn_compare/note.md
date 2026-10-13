# Token 级注意力干预：共享参数与真实推测解码对照

这轮比较三条用户选定的路线：注意力 Adapter、注意力分数偏置、直接更新小模型注意力投影。与逐位置 oracle 不同，训练共享参数，留出题目不参与拟合。

## 数据与目标

- 模型：本地 Qwen2-7B-Instruct → Qwen2-1.5B-Instruct，BF16，eager attention，单张 H100。
- 冻结的 HotpotQA 300 题文件：`exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl`。
- 训练题 cursor 0–127；校准题 128–159；测试题 200–249；划分按题互斥。
- 每个训练题最多 16 个下一 token 位置，遇到轨迹的 EOS 停止。偶数训练题沿大模型续写，奇数沿原始小模型续写；各位置目标均为大模型在同一前缀上的 argmax。
- 三轮训练，按校准集 token 命中率选择 checkpoint，相同时以交叉熵打破平局。允许选择零干预；测试集不选择权重、训练轮数或偏置强度。
- 这批数据曾用于项目的探索实验；这里的“留出”指本轮训练与校准均不使用测试题，并非外部全新测试集。

## 三条路线与对照

所有修改只作用于小模型最后四层，从 prompt 的最后一个 query 开始，贯穿生成区。更早的 prompt 表示和 KV 缓存始终由原始冻结模型生成。

1. **Token Adapter**：学习 112 条大模型注意力行到小模型 48 个头的混合权重。小型 MLP 根据当前 query 的小模型表示产生逐 token、逐头注入门控，门控最大值 0.5。混合发生在 softmax 后，大小模型主体冻结。大模型 prompt 最后一个 query 的注意力只获取一次，生成时复用；生成区新增位置的 prior 为零。它是在原有静态门控 Adapter 基础上的新变体。
2. **注意力分数偏置**：对应层的大模型头取平均，对文档 token 的 log attention 去均值、截断到 [-2,2]，乘以强度后加到小模型 softmax 前的分数。首 token、问题和生成区的偏置为零。在校准集比较 0、0.125、0.5、1.0；推理时每题只获取一次大模型 prompt 注意力。
3. **完整 Q 投影更新**：优化最后四层完整 Q 矩阵的增量，每层 1536×1536，共 9,437,184 个参数，不使用低秩分解。除下一 token 交叉熵外，增加系数 0.05 的按层、按 query 平均头注意力 KL。大小模型头数不同，通过 token 分布的平均头监督消除一一对应要求。teacher 注意力仅在训练使用，测试无需传入 teacher prior。

Adapter 学习率 0.005；Q 更新学习率 3e-5；Adam；梯度范数截断 1.0。最初的 Q 更新原型实现为 `Q = W_base h + ΔW h`，FP32 更新分支的输出转为 BF16 后相加。

随后补充 **原生 Q 权重微调**（`q_native_attention`）：训练时使用 `Q = Linear(h, BF16(W_base + ΔW))`，完整矩阵更新在原生 Q 投影内执行；推理时预先合并权重，仅执行一次原生投影，无额外矩阵乘分支。配套 `q_native_output_only` 使用同样实现，去掉注意力 KL。它们沿用相同划分、轨迹、训练预算、校准选择和评测规则，与最初增量分支分别报告，避免混淆 BF16 舍入及运行开销。两种 Q 路线都只更新解码区，prompt 前缀继续使用原始 Q 权重。

- **Adapter-uniform 对照**：相同 Adapter、初始化、训练顺序和预算，将 teacher bank 换成 prompt 可见 token 的均匀分布；不携带 teacher 注意力信息。
- **Q-output-only 对照**：相同完整 Q 参数、初始化与训练预算，只优化下一 token 交叉熵，不使用注意力 KL。

## 评测与实现检查

真实贪心推测解码：draft k=4、最多生成 32 token，teacher 一次验证整块，全部接受时追加 teacher bonus；拒绝时裁去两个模型的多余 KV，追加 teacher 校正。接受率是所有题 `accepted / proposed` 的总量比值；未接受的块内后续提议也计入分母。

另外使用测试题原始小模型续写的共同前缀，比较各路线修复原本分歧的位置数，以及破坏原本一致的位置数。此项最多每题 16 个位置，属于固定上下文评测，与真实推测解码接受率分开报告。

训练复用冻结的 prompt KV 和前 24 层表示，重新计算整个生成窗口的最后四层；窗口内各 query 保持因果关系，Q 更新对先前生成位置的影响会传播到后续位置。没有复用未经更新的生成区上层 KV。

实现检查包括零干预一致性，以及训练缓存版与完整缓存前向的多 query logits、梯度一致性。保存每题提议、接受、生成 token、校准历史和分割信息。置信区间按题配对 bootstrap 2,000 次。

时间包含 prompt prefill、teacher prior 获取、draft、验证；这是单卡 Python 原型的实测时间，不代表部署后的跨设备传输成本或优化后的吞吐。BF16 的不同 query 批长度可能改变近似并列的 argmax，因此也记录各路线与 baseline 的 teacher 输出完全一致题数。

## 运行

```bash
/home/wanghl/miniconda3/envs/rescue/bin/python exp/token_attn_compare/run_compare.py \
  --output exp/token_attn_compare/out_compare128_e3

/home/wanghl/miniconda3/envs/rescue/bin/python exp/token_attn_compare/run_compare.py \
  --output exp/token_attn_compare/out_native_q128_e3 \
  --methods q_native_attention q_native_output_only

/home/wanghl/miniconda3/envs/rescue/bin/python exp/token_attn_compare/verify_checkpoints.py \
  --output exp/token_attn_compare/out_native_q128_e3
```

完整结果见输出目录的 `summary.json`、`test_fixed_context.json`、`selection.json`、`training.jsonl` 和 `validation.json`。权重保留在本地输出目录，`.pt` 不纳入版本控制。

## 完成结果：真实块验证

128 个训练题共 1,798 个位置，32 个校准题共 418 个位置；测试共同前缀共 656 个位置。各学习路线均选中第 3 轮，偏置选择 λ=0.5。原生权重版本和增量分支版本的训练/校准输入、teacher 标签、baseline 预测完全相同。

| 方法 | 50 题实际接受率 | 相对 baseline，百分点 | 95% 配对区间，百分点 | teacher 验证块数 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 64.276% | — | — | 387 |
| Token Adapter | 63.906% | -0.370 | [-2.117, 1.006] | 390 |
| 文档 token 分数偏置 | 64.344% | +0.068 | [-0.723, 0.912] | 387 |
| 原生完整 Q 微调 + 注意力 KL | 64.586% | +0.310 | [-2.237, 2.831] | 393 |
| Adapter-uniform 对照 | 63.472% | -0.804 | [-2.447, 0.543] | 390 |
| 原生完整 Q 微调，仅输出目标 | 64.475% | +0.199 | [-2.316, 2.736] | 393 |

最初的额外 FP32 Q 增量分支：注意力 KL 版本 64.718%，输出目标对照 64.387%。这些数值也没有稳定超过 baseline。与之前 300 题实验的 60.46% 不直接比较，以本轮 50 题共同 baseline 为准。

| 方法 | 共同前缀 token 命中率 | 修复原本分歧 / 185 | 破坏原本一致 / 471 | 净增加命中 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 71.799% | 0 | 0 | 0 |
| Token Adapter | 71.494% | 5 | 7 | -2 |
| 文档 token 分数偏置 | 71.341% | 2 | 5 | -3 |
| 原生完整 Q 微调 + 注意力 KL | 75.915% | 40 | 13 | +27 |
| Adapter-uniform 对照 | 71.799% | 5 | 5 | 0 |
| 原生完整 Q 微调，仅输出目标 | 76.220% | 42 | 13 | +29 |

**结论：三条路线均已完成共享训练或校准及留出评测，但当前没有稳定的实际接受率提升。** Q 微调改善了固定前缀上的预测，却没有减少完整推测解码的验证调用。其注意力监督版本没有在固定前缀上超过输出蒸馏对照，尚不能证明 teacher 注意力提供额外收益。Q 路线参数量约 944 万，而 Adapter 为 55,004，不能把容量差异简单解释为方法优劣。

初轮 baseline/Adapter/偏置耗时分别为 39.56/41.67/43.40 秒；原生权重轮 baseline/Q-attention/Q-output-only 分别为 39.75/39.95/40.07 秒。单次单卡原型测量未观察到实际加速。

各路线的零干预，以及初始和训练后的缓存加速版/完整缓存前向 logits、梯度最大差均为 0；所选 checkpoint 的首个测试题（cursor 200）回放全部一致。所有测试题均保留，没有根据预测稳定性或实验结果筛除。

Adapter 和偏置的 teacher 输出与 baseline 50/50 一致；原生 Q 两个版本为 48/50。输出差异见 `analysis.json`，不能将这些原型结果当成严格逐比特保持 teacher 输出的证明。为消除评测目标随验证批长度变化的问题，另做固定 teacher 续写的接受率复核。

## 固定 teacher 续写复核

先以单 query 缓存解码生成每题同一条 teacher 贪心续写，再用实际小模型及其干预、KV 回滚、纠正和 bonus 规则重放草稿块。各路线均以同一组 teacher token 为验证目标；不会重新选择 checkpoint 或超参数。这里比较接受率和验证块数，不测端到端时间，因为 teacher 续写已预先获取。

```bash
/home/wanghl/miniconda3/envs/rescue/bin/python exp/token_attn_compare/run_fixed_teacher_audit.py \
  --compare-output exp/token_attn_compare/out_compare128_e3 \
  --native-output exp/token_attn_compare/out_native_q128_e3 \
  --output exp/token_attn_compare/out_fixed_teacher_audit
```

| 方法 | 固定 teacher 复核接受率 | 相对 baseline，百分点 | 95% 配对区间，百分点 | 验证块数 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 62.525% | — | — | 390 |
| Token Adapter | 62.166% | -0.359 | [-1.963, 1.039] | 393 |
| 文档 token 分数偏置 | 62.719% | +0.194 | [-0.514, 0.991] | 389 |
| 原生完整 Q 微调 + 注意力 KL | 63.673% | +1.148 | [-0.603, 3.029] | 393 |
| Adapter-uniform 对照 | 61.718% | -0.807 | [-2.441, 0.690] | 393 |
| 原生完整 Q 微调，仅输出目标 | 63.562% | +1.037 | [-0.743, 3.015] | 393 |

原生 Q 的注意力监督版本相对输出目标对照仅 +0.111 个百分点，95% 区间 [-0.311, 0.666]；真实 teacher 注意力 Adapter 相对均匀对照为 +0.448 个百分点，区间 [-0.605, 1.549]。两种注意力额外贡献都未得到支持。偏置在复核中少 1 次验证调用，变化很小；Q 的验证调用仍多于 baseline。

全部 50 题在全部路线中最终输出与固定 teacher 参考完全一致，所有总量接受率已按逐题记录复核。原始块验证 baseline 与单 query teacher 参考仅 42/50 题输出完全一致，因此绝对接受率会随缓存/验证批形状变化；这一数值敏感性足以影响当前微小增量的解释。

**复核后判断：目前 Q 微调呈现值得继续验证的趋势，但接受率增量区间仍包含零；尚无稳定接受率或加速收益，也没有证明大模型注意力优于单纯输出蒸馏。** 本轮是单模型对、单随机种子、50 个测试题，Q 仅更新解码区、训练窗口最多 16 个位置。下一轮应先固定评测目标，沿实际 teacher 校正后的推测解码前缀训练，扩大生成位置覆盖，并在更大独立测试集验证；这些后续实验尚未开展。

复核结果：`out_fixed_teacher_audit/summary.json`；逐题参考：`teacher_reference.jsonl`；检查：`verification.json`。

随后完成了 Q 可表达分数投影与正交残差补齐的诊断及固定 teacher 留出对照。数学闭合检查通过，但本轮未观察到残差补齐的稳定接受率收益；方法、参数及结果见 [geometry_note.md](geometry_note.md)。
