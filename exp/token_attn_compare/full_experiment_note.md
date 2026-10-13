# 动态 teacher 注意力与完整前缀 Q/K/V 实验

本轮对应用户指定的两组实验：静态与当前 query 的 teacher 注意力；Q、Q+K、Q+V、Q+K+V 的训练比较。以下协议在正式留出评测前固定。

## 共同协议

- Qwen2-7B-Instruct → Qwen2-1.5B-Instruct，BF16，eager attention，最后四层。
- 原始冻结 HotpotQA 数据；训练 cursor 0–127，校准 128–159，测试 200–249。三组按题互斥，但测试题曾用于项目此前探索，并非新的外部测试集。
- 单 query 缓存 teacher 贪心轨迹，最多 32 个生成 token，遇 EOS 停止。测试参考逐 token 复用上一轮 `out_fixed_teacher_audit/teacher_reference.jsonl`，准备时重新计算并逐位置核对。
- 固定目标下的实际草稿回放：draft k=4，拒绝回滚 KV，全接受追加参考 bonus。接受率是总 accepted / 总 proposed；块内首次拒绝后的其他提议计入分母。保存全部逐题块和输出。
- 校准负责选强度或 checkpoint，测试不参与选择。95% 区间按题配对 bootstrap 2,000 次，随机种子 42。
- 准备和早期校准时 GPU 有其他任务；恢复后本实验的两条路线并行运行。本轮没有端到端加速或吞吐结论，记录的原型耗时只用于进度与复现。

## 实验 A：注意力是否需要随 query 更新

只修改生成区的 student 注意力，prompt 前缀仍由原始 student 计算。干预在 softmax 前，强度 \(\lambda\in\{0,0.125,0.25,0.5,1\}\)。校准优先接受率，平局优先更少验证块，再优先较弱强度。

比较三类 prior：

1. **静态文档 prior**：每题只用 prompt 最后一个 query 的 teacher 层内头平均注意力。
2. **动态文档 prior**：每个实际草稿 query 都运行 teacher，获取同一实际前缀上的当前注意力。即使已走入将被拒绝的草稿分支，也按该分支计算，不提前读取参考 token 作为输入。
3. **动态完整 prior**：同样获取当前注意力，但覆盖文档、问题、首 token 和生成区全部可见位置。

文档干预为 \(s'_D=s_D+\lambda(\log p_{T,D}-\log\operatorname{softmax}(s_D))\)，非文档分数不变。强度 1 完全替换文档内部条件分布，并保持文档总概率质量。完整 prior 使用相同公式覆盖整个可见 token 行。teacher 目标以 \(10^{-12}\) 作概率下界，计算使用 FP32，然后将实际 attention 转为 BF16。

动态 teacher 和 student 的缓存同步前进与回滚。teacher 下一 token logits 不参与干预，参考 token 仅用于所有路线共同的验证、校正与 bonus。动态方法每个草稿 query 额外调用 teacher，是离线可行性诊断，其接受率不能当成加速收益。

正式测试同时保留：各类校准所选强度；静态/动态文档的两组同强度比较；预先规定的强度 1 对照；按题固定文档置换的动态 prior 对照。打乱仅涉及文档位置，首 token、问题、生成区保持各自原有 prior 值。重复的同一路线或零干预复用相同逐题结果，不当作独立证据。

## 实验 B：哪些投影需要更新

四组参数：Q、QK、QV、QKV。每组各做两种监督，共八条训练路线：

- **output**：\(\mathrm{KL}(p_T(\cdot\mid x_{\le t})\Vert p_S(\cdot\mid x_{\le t}))\)，温度 1，使用两个 tokenizer 共同的全部有效词表 151,646 个 token，分别重新归一化。不只使用 teacher argmax 标签。
- **relations**：输出 KL 加 0.05 倍当前注意力 KL，再加 0.05 倍 Value 关系 KL。

注意力目标是对应 teacher 层当前 query 的头平均 token 分布。Value 关系是当前输入 token 的 value 与全部可见 token value 的缩放点积，经 causal softmax 后在 KV 头间平均：\(R_V=\operatorname{mean}_{h}\operatorname{softmax}(v_{t,h}V_h^\top/\sqrt{128})\)。它对齐 token 关系，不直接复制两个模型的 value 向量；本轮同时加入两类关系监督，不单独归因于 Value 监督。

| 更新投影 | 参数量 |
| --- | ---: |
| Q | 9,437,184 |
| QK | 11,010,048 |
| QV | 11,010,048 |
| QKV | 12,582,912 |

全部使用完整矩阵 FP32 增量，前向为 `Linear(h, BF16(W_base + delta))`，推理时合并权重。原始参数保持冻结；这不是低秩 Adapter，也不是直接将原始 BF16 参数设为可训练。投影 bias、输出投影 O、MLP 和其他层冻结。

这里 Q-only 指只有 Q 投影参数可训练；上游隐藏状态改变仍会间接改变后续层的 K/V 张量，不能将它解释为整个网络的 K/V 数值全部固定。

**四种参数组合均修改完整 prompt 和生成区。** 训练只复用冻结前 24 层的表示，最后四层对完整 prompt+生成窗口重新前向并反向，prompt 中的 K/V 可获得梯度；推理每条路线重新 prefill 全部 prompt，使用自己的 KV。Q-only 也采用同样的前缀范围，因此和上一轮只改解码区的 Q checkpoint 分开报告。

每条路线 128 题 × 3 轮，teacher 轨迹最多 32 个生成位置；Adam，学习率 3e-5，梯度范数裁剪 1.0。相同种子、顺序、训练步数和数据。每轮在 32 题上执行实际草稿回放，按接受率选择 checkpoint；平局看验证块数，再看共同前缀上的输出 KL。允许选择 epoch 0。保留最终训练权重和校准所选权重。

因此所选 epoch 可以不同；接受率比较衡量各自按校准选择后的训练方案。关系监督同时含 attention 和 value 两项，不能把两条路线的差值单独归因于其中一项。

训练轨迹是 teacher 的正确续写，覆盖正确接受/校正后的前缀；没有额外收集错误草稿分支作为训练样本。QK 与 QV 参数量相同；其他组合容量不同，报告参数量并保留每组自己的 output 对照。

## 校验与运行

小规模 4/4/4 题、8-token 检查已通过：零更新 logits；非零更新时冻结下层复用版与完整模型的 logits、Q/K/V 梯度；训练后再次比较；文档前缀 K/V 确实改变；保存 checkpoint 的首题回放；全部路线输出等于同一 teacher 参考。

动态路线另检查同一当前 query 的 teacher 行与准备数据完全一致，首个 query 的静态与动态干预 logits 相同，以及每次回滚后两个模型的缓存长度正确。

```bash
/home/wanghl/miniconda3/envs/rescue/bin/python exp/token_attn_compare/prepare_full_experiment.py \
  --output exp/token_attn_compare/out_full128_data \
  --cache-dir /mnt/8T/cl/rescue_experiment_cache/full128_data

/home/wanghl/miniconda3/envs/rescue/bin/python exp/token_attn_compare/run_freshness.py \
  --prepared exp/token_attn_compare/out_full128_data \
  --output exp/token_attn_compare/out_freshness32_test50

/home/wanghl/miniconda3/envs/rescue/bin/python exp/token_attn_compare/run_qkv.py \
  --prepared exp/token_attn_compare/out_full128_data \
  --output exp/token_attn_compare/out_qkv128_e3
```

大体积缓存位于数据盘，配置、来源哈希、清单与可读结果位于项目输出目录。权重保留在本地 `.pt`，不纳入 Git。

正式 freshness 运行中途停止后，使用 `run_freshness.py --resume` 复用已经完整写出的 15 组校准与测试基线，未改变校准选择。`config.json` 中的 `recovery_history` 保留先前源码哈希；续跑改动只涉及恢复和输出持久化。

运行完成后，`audit_full_experiment.py` 核对数据与源码哈希、题目分割、全部逐题计数、固定输出、校准选择和动态 teacher 调用数。`diagnose_qkv_relations.py` 在已选 checkpoint 上测量共同测试前缀的 attention/value 关系 KL，只用于解释结果，不参与选择。`analyze_full_experiment.py` 生成完整表格、配对区间和静态图。

## 正式结果 A：注意力新鲜度

15 组校准配置与 12 条测试配置均完成。下表为主要配置；差值和区间均为百分点。

| 方法 | 强度 | 测试接受率 | 相对基线 [95% 区间] | 验证块 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 62.525% | — | 390 |
| 静态文档，校准选择 | 0.5 | 61.005% | -1.521 [-3.099, +0.082] | 395 |
| 当前 query 文档，校准选择 | 0.125 | 62.719% | +0.194 [-0.524, +0.962] | 389 |
| 当前 query 全部位置，校准选择 | 0.25 | 63.279% | +0.754 [-0.270, +1.904] | 386 |
| 上一行打乱文档位置 | 0.25 | 62.166% | -0.359 [-1.798, +1.199] | 392 |
| 当前 query 全部位置，完全替换 | 1 | 58.062% | -4.463 [-6.608, -2.279] | 410 |

新鲜度确实影响某些干预：文档强度同为 0.5 时，动态比静态高 1.386 个百分点，区间 [+0.114, +2.652]；强度 0.125 时差值为 +0.194，区间 [-0.513, +0.958]。所选动态文档和动态完整方案分别比各自打乱文档位置的对照高 1.017、1.113 个百分点，区间分别为 [+0.234, +1.977]、[+0.140, +2.234]。

这些是未经多重比较校正的探索性区间，支持“当前 query 和 token 位置包含相关信息”，但并未获得相对原始 student 的稳定提升。尤其不能把“好于打乱位置”理解为“已提高部署接受率”。动态完整强度 1 的下降说明，失配不只是 prior 陈旧；teacher 的头平均注意力与 student 的 value/输出表示未必兼容。

所选动态文档与动态完整方案分别额外运行 teacher 单 query 1,663、1,654 次。即使验证块减少，也不能抵消或忽略此成本；这里没有加速结论。所有 600 条测试输出逐 token 等于冻结参考，基线逐题回放与前一轮完全一致。

## 正式结果 B：完整前缀 Q/K/V 更新

八条路线均完成 128 题 × 3 轮（每条 384 次更新），并按各自校准结果选 checkpoint。全部复用同一 50 题、1,294 个 teacher 参考 token。差值和区间单位为百分点。

| 更新参数与监督 | 所选 epoch | 接受率 | 相对基线 [95% 区间] | 验证块 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 62.525% | — | 390 |
| Q，仅输出 | 2 | 62.567% | +0.042 [-1.603, +1.837] | 393 |
| Q，输出 + 关系 | 1 | 62.634% | +0.109 [-1.215, +1.648] | 392 |
| QK，仅输出 | 3 | 64.605% | +2.080 [-0.521, +4.757] | 385 |
| QK，输出 + 关系 | 3 | 63.630% | +1.105 [-1.486, +3.929] | 389 |
| QV，仅输出 | 2 | 63.872% | +1.347 [-0.683, +3.666] | 390 |
| QV，输出 + 关系 | 3 | 64.649% | +2.124 [-0.675, +5.069] | 388 |
| QKV，仅输出 | 2 | 64.384% | +1.858 [-0.581, +4.378] | 387 |
| **QKV，输出 + 关系** | **2** | **65.465%** | **+2.939 [+0.532, +5.287]** | **384** |

最佳点估计为 QKV 联合关系监督：944 / 1,442 个草稿 token 被接受；baseline 为 931 / 1,489。它相对同参数量、同样选中 epoch 2 的 QKV 仅输出对照高 **1.081 个百分点，区间 [+0.302, +1.971]**；相对 Q-only 关系监督高 **2.830 个百分点，区间 [+0.693, +4.993]**。本轮得到联合 token 关系监督的额外收益信号，不能只解释为多训练了 K/V 参数或多训练了一轮。

验证块总数下降 6 个，其配对 95% 区间为 [-15.025, +3.000]；10 题减少、6 题增加、34 题相同。接受率的正向区间不等于验证调用数或实际耗时已经稳定下降。

这个收益并非所有参数组合都出现：关系监督相对同组仅输出对照，Q 为 +0.067 [-0.881, +0.917]，QK 为 -0.975 [-2.224, +0.104]，QV 为 +0.777 [-0.874, +2.612]。因此不能推断补 K 或 V 任意一项就一定有效，也不能从这里断言 Attention 项或 Value 项哪一项单独贡献了收益。

共同测试前缀上的诊断也支持“目标不同”的区别：QKV 仅输出和联合关系版本的输出 KL 分别为 0.41544、0.41862，而 attention KL 为 0.57941、0.45335，Value 关系 KL 为 2.79500、2.38117。加入关系监督改善了 token 关系拟合，输出 KL 略高，实际草稿接受率却更好。两者共同前缀命中率为 85.085%、85.317%，baseline 为 83.539%；这些命中率不等于完整草稿接受率。

本轮支持优先继续 **QKV 联合 token 关系蒸馏**。这是单模型对、单种子、50 题且曾用于此前探索的测试结果；全部区间未做多重比较校正，不能称为已经稳定泛化。下一步应在新的测试题与多个种子上复现，并将 Attention、Value 关系监督分别消融，再判断是否需要进一步方法设计。本轮没有测试 O 投影更新、不同层数映射、长续写或端到端加速。

## 完成与核验

- 准备 210 题，训练/校准/测试分别有 3,411 / 839 / 1,294 个 teacher 位置；题目与 prompt 均无跨分割重复。
- 15 条校准配置、24 条训练 epoch 记录、12 + 9 条测试配置均完整，测试记录共 1,050 条（包含两个相同基线），全部生成结果等于固定 teacher 参考。
- 训练前后，冻结下层复用版与完整前向的 logits 和梯度最大差均为 0；训练后首个更新层 prompt key/value 最大变化分别为 1.83594 / 2.6875。
- 数据/源码哈希、校准选择、逐题计数、投影权重、checkpoint 回放、动态缓存回滚和 teacher 调用数均通过核验。
- 联合关系诊断与主评测的共同前缀输出指标完全一致，不参与 checkpoint 选择。

完整表格：[results.md](out_full128_data/results.md)；图：[acceptance_comparison.png](out_full128_data/acceptance_comparison.png)；核验：[artifact_audit.json](out_full128_data/artifact_audit.json)。原始结果：[freshness summary](out_freshness32_test50/summary.json)、[QKV summary](out_qkv128_e3/summary.json)、[关系诊断](out_qkv128_e3/relation_diagnostics.json)。所选权重：`out_qkv128_e3/qkv_relations.pt`；训练终点权重另存为 `qkv_relations_final.pt`，不要混淆。

结果核验与报告可复现为：

```bash
/home/wanghl/miniconda3/envs/rescue/bin/python exp/token_attn_compare/audit_full_experiment.py \
  --prepared exp/token_attn_compare/out_full128_data \
  --freshness exp/token_attn_compare/out_freshness32_test50 \
  --qkv exp/token_attn_compare/out_qkv128_e3
/home/wanghl/miniconda3/envs/rescue/bin/python exp/token_attn_compare/diagnose_qkv_relations.py \
  --prepared exp/token_attn_compare/out_full128_data \
  --qkv exp/token_attn_compare/out_qkv128_e3
# 仅报告使用已安装 matplotlib 的环境；不加载或训练模型。
/home/wanghl/miniconda3/envs/idn/bin/python exp/token_attn_compare/analyze_full_experiment.py \
  --prepared exp/token_attn_compare/out_full128_data \
  --freshness exp/token_attn_compare/out_freshness32_test50 \
  --qkv exp/token_attn_compare/out_qkv128_e3
```
