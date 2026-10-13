# 完整实验结果

指标均来自同一 50 题、1,294 个固定 teacher 输出 token，草稿长度 k=4。
差值单位为百分点，95% 区间按题配对 bootstrap 2,000 次，未做多重比较校正；属于探索性结果。

## 注意力新鲜度

| 路线 | λ / epoch | 接受率 | accepted / proposed | 相对基线，百分点 [95% 区间] | 验证块 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 62.525% | 931 / 1489 | +0.000 [+0.000, +0.000] | 390 |
| static_selected | 0.5 | 61.005% | 923 / 1513 | -1.521 [-3.099, +0.082] | 395 |
| dynamic_selected | 0.125 | 62.719% | 932 / 1486 | +0.194 [-0.524, +0.962] | 389 |
| static_at_dynamic_strength | 0.125 | 62.525% | 931 / 1489 | +0.000 [-0.932, +0.943] | 390 |
| dynamic_at_static_strength | 0.5 | 62.391% | 929 / 1489 | -0.134 [-1.544, +1.295] | 390 |
| dynamic_shuffled_selected | 0.125 | 61.702% | 928 / 1504 | -0.823 [-1.850, +0.219] | 394 |
| static_unit | 1 | 60.499% | 922 / 1524 | -2.026 [-3.596, -0.340] | 397 |
| dynamic_unit | 1 | 61.569% | 926 / 1504 | -0.956 [-2.997, +1.174] | 393 |
| dynamic_shuffled_unit | 1 | 60.433% | 921 / 1524 | -2.092 [-4.330, +0.193] | 397 |
| dynamic_full_selected | 0.25 | 63.279% | 934 / 1476 | +0.754 [-0.270, +1.904] | 386 |
| dynamic_full_unit | 1 | 58.062% | 911 / 1569 | -4.463 [-6.608, -2.279] | 410 |
| dynamic_full_shuffled_selected | 0.25 | 62.166% | 930 / 1496 | -0.359 [-1.798, +1.199] | 392 |

### 配对对照

| 比较（前者减后者） | 差值，百分点 | 95% 区间 |
| --- | ---: | ---: |
| dynamic_selected_vs_static_at_dynamic_strength | +0.194 | [-0.513, +0.958] |
| dynamic_at_static_strength_vs_static_selected | +1.386 | [+0.114, +2.652] |
| dynamic_unit_vs_static_unit | +1.070 | [-0.564, +2.792] |
| dynamic_selected_vs_dynamic_shuffled_selected | +1.017 | [+0.234, +1.977] |
| dynamic_unit_vs_dynamic_shuffled_unit | +1.136 | [-0.275, +2.626] |
| dynamic_full_selected_vs_dynamic_full_shuffled_selected | +1.113 | [+0.140, +2.234] |

## Q/K/V 训练

| 路线 | λ / epoch | 接受率 | accepted / proposed | 相对基线，百分点 [95% 区间] | 验证块 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 62.525% | 931 / 1489 | +0.000 [+0.000, +0.000] | 390 |
| q_output | 2 | 62.567% | 931 / 1488 | +0.042 [-1.603, +1.837] | 393 |
| q_relations | 1 | 62.634% | 932 / 1488 | +0.109 [-1.215, +1.648] | 392 |
| qk_output | 3 | 64.605% | 940 / 1455 | +2.080 [-0.521, +4.757] | 385 |
| qk_relations | 3 | 63.630% | 936 / 1471 | +1.105 [-1.486, +3.929] | 389 |
| qv_output | 2 | 63.872% | 937 / 1467 | +1.347 [-0.683, +3.666] | 390 |
| qv_relations | 3 | 64.649% | 940 / 1454 | +2.124 [-0.675, +5.069] | 388 |
| qkv_output | 2 | 64.384% | 940 / 1460 | +1.858 [-0.581, +4.378] | 387 |
| qkv_relations | 2 | 65.465% | 944 / 1442 | +2.939 [+0.532, +5.287] | 384 |

### 配对对照

| 比较（前者减后者） | 差值，百分点 | 95% 区间 |
| --- | ---: | ---: |
| q_relations_vs_q_output | +0.067 | [-0.881, +0.917] |
| qk_relations_vs_qk_output | -0.975 | [-2.224, +0.104] |
| qv_relations_vs_qv_output | +0.777 | [-0.874, +2.612] |
| qkv_relations_vs_qkv_output | +1.081 | [+0.302, +1.971] |
| qk_output_vs_q_output | +2.038 | [+0.036, +4.183] |
| qk_relations_vs_q_relations | +0.996 | [-1.262, +3.425] |
| qv_output_vs_q_output | +1.305 | [-0.194, +2.948] |
| qv_relations_vs_q_relations | +2.015 | [-0.239, +4.385] |
| qkv_output_vs_q_output | +1.816 | [-0.163, +3.733] |
| qkv_relations_vs_q_relations | +2.830 | [+0.693, +4.993] |
| qv_output_vs_qk_output | -0.733 | [-2.576, +1.016] |
| qv_relations_vs_qk_relations | +1.019 | [-0.663, +2.680] |

## 共同前缀输出诊断

各方法在 teacher 正确轨迹的相同位置前向；此命中率不是完整草稿回放的接受率。

| 路线 | 命中数 / 位置数 | 命中率 | 输出 KL |
| --- | ---: | ---: | ---: |
| baseline | 1081 / 1294 | 83.539% | 0.54995 |
| q_output | 1086 / 1294 | 83.926% | 0.46327 |
| q_relations | 1090 / 1294 | 84.235% | 0.49308 |
| qk_output | 1098 / 1294 | 84.853% | 0.43585 |
| qk_relations | 1098 / 1294 | 84.853% | 0.43607 |
| qv_output | 1104 / 1294 | 85.317% | 0.43077 |
| qv_relations | 1102 / 1294 | 85.162% | 0.42485 |
| qkv_output | 1101 / 1294 | 85.085% | 0.41544 |
| qkv_relations | 1104 / 1294 | 85.317% | 0.41862 |

## 共同测试前缀的关系对齐

KL 越低表示更接近 teacher 的头平均 token 关系；这些诊断不参与 checkpoint 选择。

| 路线 | Attention KL | Value 关系 KL |
| --- | ---: | ---: |
| baseline | 0.51304 | 2.60000 |
| q_output | 0.52075 | 3.12365 |
| q_relations | 0.46967 | 2.64314 |
| qk_output | 0.59928 | 2.67892 |
| qk_relations | 0.45649 | 2.53257 |
| qv_output | 0.53449 | 3.13188 |
| qv_relations | 0.45150 | 2.32029 |
| qkv_output | 0.57941 | 2.79500 |
| qkv_relations | 0.45335 | 2.38117 |

## 动态注意力的额外成本

| 路线 | teacher 额外单 query 调用 | 原有验证块数 |
| --- | ---: | ---: |
| baseline | 0 | 390 |
| static_selected | 0 | 395 |
| dynamic_selected | 1663 | 389 |
| static_at_dynamic_strength | 0 | 390 |
| dynamic_at_static_strength | 1664 | 390 |
| dynamic_shuffled_selected | 1678 | 394 |
| static_unit | 0 | 397 |
| dynamic_unit | 1676 | 393 |
| dynamic_shuffled_unit | 1694 | 397 |
| dynamic_full_selected | 1654 | 386 |
| dynamic_full_unit | 1732 | 410 |
| dynamic_full_shuffled_selected | 1669 | 392 |

调用数用于揭示 oracle 成本，不能直接折算成 wall-clock 加速；本轮并行运行，原型计时不作性能比较。

Q/K/V 推理只需合并后的 student 权重，teacher 关系只用于训练。训练方案均为单种子；模型层数相同，本轮不检验不同层数的对应关系。
