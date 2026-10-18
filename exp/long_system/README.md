# 长上下文RESCUE系统优化

范围：先优化RESCUE，并与SpecExtend比较。未运行EAGLE-3、DFlash等其它性能基线。
当前协议和状态见[PLAN.md](PLAN.md)。在完整矩阵完成前，不能称已超过SOTA。

## 实现

- `rescue.efficient_attention`：Qwen2高效SDPA、按需Q快照，缓存多query保持右下因果
  对齐；FP32 GQA通过KV头折入batch使用memory-efficient内核。
- `rescue.selection`的`hierarchical`：粗chunk候选、细token覆盖。内容按已提交位置
  归一化memo，覆盖目标限于高联合注意力支持集，矩阵行数有预算上限。这改变了
  原全历史覆盖目标，是近似算法，不是无损实现等价。需要匹配预算消融。
- 自适应刷新：上一已验证注意力的总变差与相对覆盖保留量触发重选，带最小/最大
  间隔。不使用未验证Teacher query，不使用未来接受结果做当前选择。
- `rescue.rollout_training`：真实SD状态中的末层mapper重放，冻结大小模型，用
  输出KL、Teacher argmax CE与归一化残差惩罚；不对不可导接受率直接求梯度。
- `attention_injected`长度模型：原注意力/读取特征加首token实际注入摘要，第二个
  token前决定长度。可本地串行/异步，36维特征尚未接入旧31维TCP协议。
- `rescue.packing`：可选不可变QKV和MLP gate/up权重打包。应在最终设备/精度上
  安装，安装后不再移动模型。比较时同样应用到控制组，记录准备成本和额外显存。
  同权重架构wrapper顺序执行时共享打包操作，避免复制整套gate/up权重；该共享
  对象保存当前forward的QKV view，不是并发多请求服务接口。

历史默认FP32/eager和原mapper/31维风险checkpoint仍可运行。混合prefill、高效
注意力、分层选择、投影打包和新长度checkpoint均为显式选项，未静默替换默认。

```bash
python -m rescue --prompt-file prompt.txt --verify --efficient-attention \
  --prefill-precision bfloat16 --pack-projections \
  --selector hierarchical --coverage-rows 64 \
  --injection-layers 27 --inject-tokens 64 --length-policy fixed --fixed-k 4
```

BF16 prefill改变前缀数值，使用同精度独立Target参考；不声称与旧FP32完整生成位级
一致。第一轮全BF16冒烟4K出现参考分歧，失败记录保留；不能用它声称无损加速。

## SpecExtend对照

1. 轻量Qwen2 chunk检索：同骨架、32-token chunk、Top16/32、每4轮刷新，固定
   长度在开发集选定。它不是完整原版SpecExtend。
2. 完整树算法流Qwen2适配：沿用归档的树构造、跨模型检索、Hybrid Tree Attention；
   适配QKV bias、RoPE基数、GQA缓存头数/精度和可用词表。共享Target prefill，
   FP32前缀输出/LSE使用PyTorch efficient SDPA，Draft保留原窗口512 Flash prefill。
   这是同模型可比的系统移植，**不是原版Vicuna/FP16论文数字复现**。

适配器不修改原归档源码，完整引擎运行使用独立`specenv`依赖。词表padding部分
在两个引擎中均排除。所有模型在同一H100；没有跨边云模型部署性能声明。

```bash
python -m rescue.long_data
python -m rescue.long_train --phase collect --output exp/long_system/new_training \
  --cache /path/to/new_tensor_cache --max-new-tokens 256 --prefill-precision bfloat16
python -m rescue.long_train --phase train --output exp/long_system/new_training
# 使用具有flash-attn依赖的独立环境运行完整对照
python -m rescue.long_benchmark --training exp/long_system/new_training \
  --output exp/long_system/new_comparison --pack-projections
OMP_NUM_THREADS=4 python -m rescue.long_audit exp/long_system/new_comparison
OMP_NUM_THREADS=4 python -m rescue.long_training_audit exp/long_system/new_training
```

数据准备要求本地原始GovReport parquet，来源和SHA256在[data/manifest.json](data/manifest.json)。
自然报告截断成嵌套2K/4K/8K/16K，不重复填充；train/development/test文档哈希
互斥。生成与匹配精度Target参考核对，不用完整报告摘要评分截断前缀的回答。
模型和训练张量是本地ignored工件；脚本、协议、训练记录与原始JSON计时可归档。

当前实际采用两阶段开发：初轮114请求完成后先延后独立测试，再补齐50/64树
节点与优胜预算的无注入控制。补充运行`rescue.long_refine`，最终保存记录使用
`rescue.long_refine_audit`核验；测试结果未用于两阶段开发选择。
