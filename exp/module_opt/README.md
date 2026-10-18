# 面向模块指标的优化

本轮已完成，结果见[完整报告](REPORT.md)。M1新锚点候选未改善，保留原方案；
M2 query向量头有小幅纠错信号，读取条件补充未进一步改善；M3 ratio能少浪费，
但增加验证、没有净加速。180个确认输出一致，52项CPU检查通过。确认只有三篇
开发文档，不构成完整方法成功或跨域泛化证据。

本轮针对上一轮[模块目标报告](../module_objectives/REPORT.md)的失败目标继续优化，
不恢复原来71/320的独立测试，不扩展外部SOTA对照。

- M1：给分层覆盖增加高联合注意力锚点与非负注意力质量项，避免覆盖增益饱和后
  保留任意低注意力位置。比较原512、锚点50%/75%、256预算锚点75%；训练侧先
  要求验证轮数不超过原512，再按完整请求TPS选择，不能只靠减少参考数宣称有效。
- M2：query条件低秩向量头改变V修正方向，原架构只改变标量强度。零初始化保留
  原提议；缓存静态仿射，解码时仅算query项。选择M1后重新收集真实滚动状态；
  仅在拟合文档生成Target已知的有界V修正辅助标签，再训练向量头及末层gate。
  监测query不生成oracle标签；替换下游mapper必须增加监测贪心命中，KL只打破平局。
- M3：在实际采用的mapper下重新收集风险状态，训练排除A·V读取误差的轻量注意力
  风险器；比较注意力先验＋请求内已验证条件反馈、纯历史反馈、ratio与提交数减
  训练速率乘成本的surplus目标。反馈止于首个拒绝/EOS，新请求重置，不看未来标签。

模型/精度/设备沿用Qwen2-7B/1.5B、H100、Target FP32 decode/混合BF16 prefix，
Draft BF16，TF32关闭，共同高效注意力与投影打包。完整backing KV仍保留。

训练拟合211/213/220/226/233，训练监测236，选参271/272/288，预热246；输入
8K/16K，输出上限128。确认使用原GovReport validation中索引300之后首次满足
自然16K长度且与此前long_system三个划分的文档SHA不重叠的303/306/327，参数冻结后才打开生成。
这仍是三文档开发确认pilot，不是跨任务论文确认或SOTA证据。

初始协议和源码在`out_v1_20261018/protocol.json`与`source/`；M1/M2/M3分别保存
选参/训练记录，`locked.json`在确认前冻结，确认计时含两次随机顺序重复。
训练profile/采集不提供速度结论；性能比较使用无profile、无collector的完整请求。
训练tensor与`.pt`存放在本地ignored缓存，精确重放需要这些二进制文件。

```bash
python -m rescue.module_opt_data --output exp/module_opt/new_data
python -m rescue.module_opt_run --data exp/module_opt/new_data \
  --cache /path/to/new_tensor_cache --output exp/module_opt/new_run
```

默认旧行为不变，新锚点权重默认0；query mapper与轻量/反馈策略需要显式checkpoint。
保存记录核验命令（不启动GPU实验）：

```bash
OMP_NUM_THREADS=4 python -m rescue.module_opt_report exp/module_opt/out_v1_20261018
OMP_NUM_THREADS=4 python -m rescue.module_key_report exp/module_opt/out_key_v1_20261018
python exp/module_opt/audit_saved_runs.py
```

前两项需要当前环境中被忽略的checkpoint；跨阶段核验还核对本地采集tensor哈希。
Git文本记录含原始请求和统计，但不打包模型/tensor二进制。CPU检查见`cpu_checks.txt`。

方向头在训练监测中未增加命中，因此另在确认输出打开前冻结了
`key_attention_plan.json`：加入当前Student Q/K给出的每KV-head读取权重，并按实际
attention输出效果蒸馏FIT oracle修正。该M2-only实验使用相同确认文档，不把它们
算成另一组独立样本；输入是Teacher V、Student V/query及Student读取权重，没有
声称注入Teacher K或验证独立的逐head硬门控。入口为`rescue.module_key_run`。

训练侧还显示M3的两个折中端点：ratio反馈废弃408/验证287，surplus反馈504/255，
固定4为501/258。`length_pareto_plan.json`据此补充两种端点的匹配确认，避免仅按
TPS选中的surplus掩盖少浪费的ratio；它们都使用原mapper，不与keyed mapper合并。
