# CSGB 单轮接受量收益实验归档

本目录保存已完成的 4K CSGB 内部实验：相对训练集选择的 `fixed8_native`，平均每轮接受的 Draft Token 增加 **0.3013**，配对文档 95% CI 为 **[0.1684, 0.4070]**；完整请求 TPS 下降 **2.35%**，95% CI 为 **[−4.56%, −0.79%]**。归档保留这项接受量收益，不将它描述为端到端加速。

这是 **Native-V 的候选树节点预算控制**实验。Mapper 未启用，最大深度固定为 10，控制器选择整棵树的节点预算 `{8,16,24,32}`。它不是后续的固定 V-only Mapper 加动态最大深度方案，也没有测量该方案的联合收益。

## 结果与对照

| 指标 | 相对 fixed8_native 的配对差值 | 95% CI |
|---|---:|---:|
| 接受长度，Draft Token 每轮 | +0.301296 | [+0.168362, +0.407044] |
| 完整请求 TPS | −2.345948% | [−4.558755%, −0.790447%] |

独立统计单位是 8 个历史训练内监测文档。每个文档先对同一动作流的两次计时取中位数，再平均三个动作流，最后做配对文档 bootstrap。三个动作流不等于三个独立训练种子。

常量输入控制器也有近似的接受量增益；CSGB 相对固定16节点的接受长度差异区间跨零。因此这里的正结果仅说明相对固定8节点接受量增加，不能据此断言注意力特征具有独立贡献，或优于所有固定预算。

- [完整原始报告](analysis/REPORT.md)
- [统计结果](analysis/summary.json)
- [接受长度图](analysis/acceptance.png) 与 [SVG](analysis/acceptance.svg)
- [完整请求 TPS 图](analysis/request_tps.png) 与 [SVG](analysis/request_tps.svg)
- [观测预算分布](analysis/observed_budget_map.png)
- [特征与决策耗时](analysis/decision_latency.png)
- [训练轨迹](analysis/training_learning_curve.png)
- [动作流波动](analysis/action_stream_variability.png)

## 算法和固定设置

模块1提供上一轮验证的 Target 注意力、下一轮 Root 实际可读位置，以及构造缓存时的注意力快照。控制器计算 Coverage、Cache-Anchored Shift 及两者乘积，输入为 `[1,C,D,C×D]`。该历史版本每轮观测注意力，但缓存选择仍每4轮刷新。

控制器是很小的 softmax 线性策略，用完整请求奖励 `N − rho × T` 训练，每个请求结束后才更新；监测时参数冻结，按保存的独立动作流采样。它不是 LinUCB，也不是尚未运行的深度收益预测器。

| 固定项 | 设置 |
|---|---|
| Target / Draft | Vicuna 7B 16K / Vicuna 68M |
| 实际输入长度 | 4096 Token |
| 输出上限 | 128 Token |
| 工作缓存选择 | Attention Top32，Chunk 32 Token，每4轮刷新 |
| Target / Draft 精度 | eager FP32 / FP16，TF32 关闭 |
| 硬件 | 服务器报告的 48GB RTX 4090 |
| 训练与内部监测 | 历史文档0–15 / 16–23，单训练种子101 |
| 请求数 | 64校准＋96训练＋336监测＋60预热＝556 |
| 输出核验 | 556个请求均与对应 Target AR 参考 Token 一致 |

完整请求计时包含 prefill、实际注意力与特征开销、Draft、Target 验证及结束等待；不包含独立 AR 参考、请求外训练更新和结果序列化。这是受控实现测量，不是生产后端速度。8K、独立确认集和 Mapper 联合方法不在本实验范围内。

## 代码入口

`source/` 保存执行时冻结的52份源码及对应配置，而不是从当前开发工作区挑出的最新版。按仓库规则不上传过程日志；其原始哈希仍保留在清单中，其余原文件字节不变。

- [策略及梯度](source/specextend/csgb_episode_policy.py)
- [逐轮控制与缓存时序](source/specextend/csgb_episode_harness.py)
- [原实验 Runner](source/specextend/csgb_episode_runner.py)
- [Coverage 与 Shift](source/specextend/csgb.py)
- [真实 Root 可读位置](source/specextend/csgb_cache_geometry.py)
- [解码引擎](source/specextend/classic/model_classic.py)
- [冻结控制器参数](frozen/frozen_policies.json)
- [原执行协议](frozen/protocol.json)
- [原分析源码](analysis/analysis_source.py)
- [独立绘图源码](analysis/figure_source.py)
- [完整原始结果压缩包](raw_results.tar.gz)

冻结参数包含 CSGB、constant、history 三个对照的权重、缩放、温度和预算。没有上传基础语言模型权重、凭据或本次暂停的深度控制器半成品。

## 本地核对和重画

只核对归档及保存结果，不需要 GPU：

```bash
python verify_archive.py
```

安装 NumPy 后，核对脚本还会从原始请求重新计算主要配对置信区间；缺少 NumPy 时明确报告未重算 CI，不冒充完成该项。

重画六组图需要 NumPy 和 Matplotlib，且输出目录必须尚不存在：

```bash
python -c 'import gzip,pathlib; pathlib.Path("plot_data.json").write_bytes(gzip.decompress(pathlib.Path("analysis/plot_data.json.gz").read_bytes()))'
python analysis/figure_source.py --data plot_data.json --out rerendered
```

`source/queue_csgb_episode_pilot_v3.sh` 保留原服务器路径、GPU身份检查、防覆盖和前序证据检查，**不要直接用它重跑或覆盖现有服务器实验**。在其他机器重新做 GPU 实验，需要独立输出目录、依赖环境、合法取得的基础模型和显式的路径及来源配置。基础模型的原文件哈希保留在协议中；本归档不声称脱离原环境即可精确重现 GPU 数值与计时。

## 来源与范围

原始运行目录为 `CSGB_episode_pilot_20261016_v3`，结果分析的原始 SHA 和服务器路径保持历史含义。[ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json) 额外绑定本次发布文件；[原完成核验](provenance/pilot_completion_verification_v1.json) 记录独立核对结果。

本归档按单轮接受量的正向信号保留该方案。原实验的速度继续门槛未通过，报告和图中的 PARTIAL 指它仍只是完整4K＋8K研究中的内部试验，并非缺失本轮336个监测请求。
