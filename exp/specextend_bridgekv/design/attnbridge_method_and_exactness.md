# AttnBridgeKV：注意力引导的跨模型 KV 协作投机解码

## 1. 方法定位

AttnBridgeKV 仍属于投机解码：小模型产生 proposal，大模型逐步验证并决定最终提交的 token。变化的是协作接口不再只有“候选 token”，还加入了只服务于 proposal 的近似共享状态。

方法维护三类缓存：

1. `C_T`：大模型的 canonical KV，只由已经提交的 token 写入；验证器始终读取它。
2. `C_D`：小模型原生 KV。
3. `C_B`：由大模型已提交前缀经映射得到、只供小模型读取的 BridgeKV。

核心创新是**注意力引导**：大模型在已提交前缀上的注意力分数决定哪些历史块进入固定预算，小网络再把这些位置的大模型状态映射成小模型缓存格式。BridgeKV 可以改变小模型 proposal，但不能写入 `C_T`，也不能绕过 token verifier。

可编辑结构图见 [bridgekv_exact_speculative_decoding.drawio](bridgekv_exact_speculative_decoding.drawio)。

## 2. 当前 4090 原型

设大模型末层 value cache 在位置集合 `S_t` 上为

\[
V_T^L[S_t]\in\mathbb{R}^{|S_t|\times d_T}.
\]

固定块预算为 `K`，块得分由大模型对已提交路径的注意力聚合得到：

\[
s_j=\operatorname{Agg}_{h,q}\left(
\frac{1}{|B_j|}\sum_{i\in B_j}A^T_{h,q,i}
\right),\qquad
S_t=\operatorname{TopKBlocks}(s,K),
\]

并强制保留因果前沿的最新块。映射器为共享低秩编码器加逐层解码头：

\[
z=\operatorname{SiLU}(W_e\operatorname{LN}(V_T^L[S_t])),
\qquad
[\widehat K_D^\ell,\widehat V_D^\ell]=W_d^\ell z.
\]

原型用插值做机制验证：

\[
K_D^{\ell\prime}=(1-\alpha)K_D^\ell+\alpha\widehat K_D^\ell,
\qquad
V_D^{\ell\prime}=(1-\alpha)V_D^\ell+\alpha\widehat V_D^\ell.
\]

另有 value-only 变体，只替换第二式而保留小模型原生 `K_D`。这是重要消融，因为缓存中的 key 带有位置相关旋转；不显式建模位置时，直接映射 key 往往比 value 更脆弱。

当前实现是 **delta/blend 原型**：仍先计算完整的小模型 KV，再混合映射结果。它可以验证注意力选块、跨模型状态对齐和 proposal 质量，但尚不能证明省掉 draft prefill 后的真实加速。下一阶段的 replacement 版本需要增量生成 draft-native key（再应用小模型的位置变换）和 value，并跳过相应的小模型 prefill。

## 3. 为什么最终分布可以保持不变

在某一已提交历史 `h` 下，令大模型的精确下一 token 分布为 `p(x|h)`，使用注意力引导和 BridgeKV 后的小模型 proposal 分布为 `q_φ(x|h,C_B)`。单 token 的精确投机采样采用

\[
a(x)=\min\left(1,\frac{p(x|h)}{q_\phi(x|h,C_B)}\right)
\]

作为接受概率；拒绝时从残差分布

\[
r(x|h)=\frac{[p(x|h)-q_\phi(x|h,C_B)]_+}
{\sum_y[p(y|h)-q_\phi(y|h,C_B)]_+}
\]

采样。proposal 被接受并输出 `x` 的概率为

\[
q_\phi(x)a(x)=\min(p(x),q_\phi(x)).
\]

拒绝事件的总概率是

\[
R=1-\sum_y\min(p(y),q_\phi(y))
=\sum_y[p(y)-q_\phi(y)]_+.
\]

拒绝校正对输出 `x` 的贡献为

\[
Rr(x)=[p(x)-q_\phi(x)]_+.
\]

所以最终边缘分布恰为

\[
\Pr(X=x)=\min(p(x),q_\phi(x))+[p(x)-q_\phi(x)]_+=p(x).
\]

这个恒等式对任意 `q_φ` 成立，因此 BridgeKV、注意力选块和映射网络可以任意改变 proposal，只要同时满足：

- verifier 的 `p` 由未修改的 `C_T` 精确计算；
- 接受率使用实际 proposal 概率 `q_φ`；
- 拒绝时使用精确残差校正；
- BridgeKV 只依赖已经提交的历史，不能偷看将要生成的 token；
- 只有通过验证的 token 才能推进 canonical cache。

多 token/树形投机解码可在每个条件历史上重复上述恒等式，因此按归纳法保持与目标模型相同的联合分布。当前 4090 代码运行的是 greedy verifier；其可检查性质是：对固定请求长度，所有策略的输出 token ID 应与原生 verifier 路径逐位相同。它不是对随机采样分布不变的经验替代。

## 4. 训练与系统目标

只最小化 KV 重构误差不足以保证系统收益。建议联合目标写成

\[
\mathcal L =
\lambda_{\text{cos}}\mathcal L_{\text{cos}}
+\lambda_{\text{nmse}}\mathcal L_{\text{nmse}}
+\lambda_{\text{KL}}D_{\text{KL}}(p\|q_\phi)
-\lambda_A\Delta A_{\text{cf}}
+\lambda_T T_{\text{e2e}}.
\]

其中 `T_e2e` 必须是真实端到端时间，而不是只看 mapper kernel；`ΔA_cf` 是配对反事实接受收益。注意力创新的主对照为

\[
\Delta A_{\text{attn}}
=A(\text{attention selection}+\text{same Bridge})
-A(\text{recent selection}+\text{same Bridge}),
\]

共享状态对齐的主对照为

\[
\Delta A_{\text{align}}
=A(\text{aligned Target-V})
-A(\text{shuffled Target-V}).
\]

两组对照都固定 mapper、`α`、块数、块大小、prompt、随机种子和生成长度；文档是配对统计单元。

近似吞吐可理解为

\[
\operatorname{TPS}\approx
\frac{\mathbb E[\text{每轮提交 token 数}]}
{T_{\text{draft}}+T_{\text{map}}+T_{\text{retrieve}}+T_{\text{verify}}},
\]

但最终结论必须使用实测 wall-clock token/s。接受长度上升并不自动等于吞吐上升。

## 5. 不能越过的结论边界

- 大模型验证的是候选 token，不是“BridgeKV 本身正确”。
- 传一个 token ID 通常远小于传一段 KV；本方法的潜在速度来源是减少重复 prefill、提高接受率或隐藏计算/通信，而不是“KV 天然比 token 更省带宽”。
- low-rank mapper 不等于 low-memory cache；若最终物化所有 draft 层的 dense K/V，缓存占用仍接近原生 draft KV。
- delta/blend 原型没有跳过 draft prefill，不能据此声称共享 KV 已带来系统加速。
- 当前同架构 Vicuna/LLaMA 结果不能直接外推到 tokenizer、层数或注意力结构不同的任意异构模型。

## 6. RTX 4090 第一阶段结果

完整统计与图见 [AttnBridgeKV 4090 实验报告](../results/rtx4090_reproduction/attn_bridgekv_pilot/REPORT.md)。实验用 4 个独立文档采集 10,669 个映射对，另外 8 个未参与训练的 GovReport 文档用于 4,096/8,192-token 成对评测；每个单元生成 128 token。mapper 在文档分组留出集上的最终 cosine 为 0.8667，NMSE 为 0.2475。

主要配对结果（文档 bootstrap 95% CI）为：

| 对照 | 4K Δ接受长度 / Δtoken/s | 8K Δ接受长度 / Δtoken/s |
|---|---:|---:|
| 注意力 Bridge − 近期 Bridge（同为 `α=0.25`） | +0.354 `[+0.209,+0.490]` / +4.49 `[+2.43,+6.46]` | +0.182 `[+0.097,+0.295]` / +1.28 `[+0.60,+2.17]` |
| 对齐 Target-V − 打乱 Target-V | +0.084 `[+0.019,+0.158]` / +1.11 `[+0.10,+2.30]` | +0.007 `[-0.031,+0.044]` / +0.05 `[-0.24,+0.33]` |
| value-only − K+V（同为注意力、`α=0.25`） | +0.377 `[+0.204,+0.570]` / +4.20 `[+1.86,+6.82]` | +0.151 `[+0.077,+0.242]` / +0.89 `[+0.32,+1.61]` |

这些结果支持两个有限结论：

1. **注意力引导本身有效。** 在相同 mapper、混合强度和缓存预算下，注意力选块稳定优于近期选块；这是当前最强的创新点证据。
2. **当前 key 映射是主要短板。** value-only 明显优于 K+V 混合，但仍未超过原生 attention KV；4K 上 value-only 相对原生注意力的接受长度差为 -0.101（CI 跨 0），吞吐差为 -1.44 token/s（CI 不跨 0）。因此现阶段不能声称 Bridge 已改善原生 proposal 或端到端性能。

精确性方面，主对照 `attention_bridge_a25` 与 `recent_bridge_a25` 在 4K/8K 的所有文档上都与修复后的 target-only AR 参考完整 128-token 一致。全部 128 个策略–文档单元中有 3 个辅助单元不一致；它们都集中在 4K 的同一文档、输出位置 15。该位置 canonical target 的 top-1/top-2 logit margin 只有 0.015625，说明 q_len=1 AR 与 batched tree attention 的 FP16 舍入在近并列处选择了不同 argmax。切换到精确 target attention 已把快速 Flash/Hybrid verifier 的 9 个不一致降到 3 个，但还不能宣称有限精度下的全局 bit-exact；需要加入低 margin 单步复核，或统一 canonical 数值内核。

最后，当前 profiler 统计的是同 GPU 上重复读取/物化的逻辑字节，不是网络传输。机制图显示每次运行累计读取数百 MiB Target-V，因此这些数据不能支持“传 KV 比传 token 更快”的说法。真正的系统论证必须在 replacement 版本中测量跳过 draft prefill 后的端到端收益及实际链路字节。
