python3 length_sweep_attention_decode_eval_maxL_fixed_v2.py \
  --model_path /home/ztl/jwt/models/Qwen2-1_5B_Instruct \
  --dataset_name hotpotqa \
  --dataset_config distractor \
  --dataset_path /home/ztl/jwt/datasets/hotpotqa/hotpot_qa/distractor \
  --dataset_split validation \
  --max_samples 10 \
  --doc_mode all \
  --max_L 2050 \
  --max_new_tokens 32 \
  --dtype bfloat16 \
  --device_map auto \
  --output_dir rag_kv_decode_eval_out \
  --enable_postprocess \
  --debug_print_first 10

  --debug_prompt_jsonl_name

python3 length_sweep_attention_decode_eval_maxL_fixed_v2.py \
  --model_path /home/ztl/jwt/models/Qwen2-1_5B_Instruct \
  --dataset_name hotpotqa \
  --dataset_config distractor \
  --dataset_path /home/ztl/jwt/datasets/hotpotqa/hotpot_qa/distractor \
  --dataset_split validation \
  --max_samples 500 \
  --doc_mode all \
  --max_L 2050 \
  --max_new_tokens 32 \
  --dtype bfloat16 \
  --device_map auto \
  --output_dir rag_kv_decode_eval_out \
  --enable_postprocess \
  --debug_print_first 500

如果你还想避免出现“回答完又开始写下一题 Question”，再加：
  --stop_on_question

1. 修改了加载数据集
2. 调整了丢弃逻辑
3. --debug_print_first 2 可以支持打印一部分L来debug（重要）
4. 加了一组对生成的答案做微处理的指标来更好的衡量--enable_postprocess \
5. 微处理后的图也调好了

1) 为什么同样的文档，L_A 会明显大于 L_B？
因为 Route A 和 Route B 的“被 token 化的文本”并不完全相同。
Route A 的长度 L_A 来自把所有 doc + query 拼成一个总 prompt：
full_prompt = make_full_prompt(doc_texts, query)，然后整体 tokenize 得到 full_ids，L_A = len(full_ids) 
length_sweep_attention_decode_e…
这里的拼接模板是：每个 doc 前都会加上类似 "[Doc i]\n" 的前缀，并且 doc 与 doc / doc 与 query 之间用 "\n\n" 分隔 
length_sweep_attention_decode_e…
Route B 的长度 L_B 则是：每个 doc 单独 tokenize（不带 [Doc i] 标头，也不带分隔符），再加上 query 的 token 数：
L_B = sum(len(doc_i_ids)) + len(query_ids) 
length_sweep_attention_decode_e…
所以你看到同一条样本里 L_A=1275, L_B=1224 是合理的：
Route A 多出来的 token，主要来自：
每个 doc 的 "[Doc i]\n" 这类标头 token
doc/doc、doc/query 之间的 "\n\n" 分隔符 token
以及整体拼接后，tokenizer 的 BPE 切分可能略有变化（少量差异）
结论：你现在的 L_A vs L_B 差异，本质是“模板差异”，不是“文档缺失”。


2) 文档到底有没有被“正常无误拼接上去”？还能不能打印前两个例子到终端？
能，而且你现在这份脚本仍然支持。
你在主循环里已经保留了 --debug_print_first：当它 >0 时，会打印前 N 个样本的：
num_docs
L_A / L_B / L
以及 full_prompt 的前 240 个字符预览（把换行显示成 \n）
length_sweep_attention_decode_e…
也就是说你可以这样跑来验证“拼接文本是否正常”：
python3 length_sweep_attention_decode_eval_maxL_fixed_v2.py \
  ... \
  --debug_print_first 2
它打印的是 Route A 的 full_prompt（也就是“真正拼接后的文本”），因为 full_prompt 就是在 pretokenize_for_routes() 里由 make_full_prompt() 生成并返回的 
length_sweep_attention_decode_e…
但注意：当前 debug 只打印前 240 个字符。如果你想看“完整拼接文本”，我建议你再加一个很轻量的功能：把 full_prompt 写到文件（比如 output_dir/prompt_debug_{i}.txt），这样你能逐字检查 doc 的拼接顺序、分隔符、是否混入奇怪字段等。你要的话我也可以直接把这段 patch 给你加进脚本。

3) 其他 RAG 实验里，文档（chunk）都是直接拼到问题前面吗？
最常见的 baseline 是的：把检索到的 passages/chunks 直接 “stuffing” 到问题前面，然后让模型一次性回答（你现在的 Route A 就属于这一类）。典型格式就是：
Context: <doc1>\n<doc2>\n...
Question: ...
Answer:
不过更成熟的 RAG 系统通常还会做一些“文档处理”，不只是简单拼接：
排序 / 重排（rerank）：先检索很多段，再挑 top-k 最相关的拼进去（减少噪声、减少长度）。
压缩 / 摘要（context compression）：对每个 chunk 做一句话压缩或抽取关键句，再拼接（更省 token）。
分段推理（map-reduce / iterative）：先对每段各自产生局部答案/证据，再汇总生成最终答案。
强约束输出格式：例如强制模型只输出 “yes/no/实体名/年份”，降低你遇到的“正确答案+冗余乱跑”的情况（你前面做的 lightweight postprocess，就是在评测层面缓解这个问题）。


4) 轻量说明：这次新增/修改的逻辑是什么？
A/B 分隔符与 doc 格式统一（控制变量）
A 的 doc 块是：[Doc i]\n{doc}，doc 与 doc 之间 \n\n，docs 与 query 之间也是 \n\n。
B 现在每个 doc 的独立 prefill 输入也变成：[Doc i]\n{doc}\n\n（注意最后一个 doc 后面也补 \n\n），这样 B 的 docs 部分 tokenization 与 A 完全一致，只是注意力隔离 + position 重置不同。
新增一个 prompt-dump JSONL 文件（不再控制台打印 prompt 预览）
--debug_print_first n 会在 output_dir/debug_prompts.jsonl（可用 --debug_prompt_jsonl_name 改名）写入 2n 行：
route=A：保存 full prompt
route=B：保存“docs_part + query”的拼接 prompt（用于对照）(这是第七版)