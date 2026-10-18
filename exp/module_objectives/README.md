# 模块目标与调度诊断

用户最新要求及指标定义见[MODULE_OBJECTIVES](../../docs/MODULE_OBJECTIVES.md)。
原长文本独立测试保持暂停，本轮使用保存训练/开发记录及新的开发pilot。
本轮已完成，结论和边界见[REPORT.md](REPORT.md)。132/132系统输出一致，
同前缀质量诊断及调度/模块开关的保存记录核验通过；当前仍无完整方法成功结论。

- `out_audit_v2_20261018/existing_accounting.json`：匹配配置的工作量重算。
- `out_audit_v2_20261018/horizon_utility.json`：训练内监测的长度目标标定；单起点
  成本模型分析，不是完整滚动TPS。v1是已被目标修订替代的原型，不用于新GPU运行。
- `out_study_v1_20261018`：新mapper目标训练、共同前缀质量、匹配模块/调度运行。
  执行协议冻结后才打开新的三篇开发文档；不读取原暂停测试的计时或输出。

```bash
OMP_NUM_THREADS=4 python -m rescue.objective_audit --output exp/module_objectives/new_audit
python -m rescue.module_study --utility exp/module_objectives/new_audit \
  --output exp/module_objectives/new_study
```

原始数据/模型路径沿用长文本环境。训练张量和`.pt`为本地ignored工件，精确重放
需要这些二进制文件，仓库内JSON报告不能替代它们。旧冻结源码和暂停记录不改写。
