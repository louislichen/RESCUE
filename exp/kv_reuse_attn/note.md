--实验组，all,300,一个全重，一个chunk，看注意力会不会发散，
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python length_sweep_attention_fixed.py \
  --model_a_path /home/ztl/jwt/models/Qwen2-7B-Instruct \
  --dataset_name hotpotqa \
  --dataset_config distractor \
  --dataset_path /home/ztl/jwt/datasets/hotpotqa/hotpot_qa/distractor \
  --split "train[:50%]" \
  --max_samples 300 \
  --dtype bfloat16 \
  --doc_mode all \
  --max_L 2050 \
  --output_plot kvreuse_metrics.png \
  --plot_dpi 800


--对照组，all，全部重算，理论上应该自己和自己都一样，散度0，皮尔森1
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python length_sweep_attention_fixed.py \
  --model_a_path /home/ztl/jwt/models/Qwen2-7B-Instruct \
  --dataset_name hotpotqa \
  --dataset_config distractor \
  --dataset_path /home/ztl/jwt/datasets/hotpotqa/hotpot_qa/distractor \
  --split "train[:50%]" \
  --max_samples 100 \
  --dtype bfloat16 \
  --doc_mode all \
  --control_mode recompute_self \
  --max_L 2050 \
  --output_plot kvreuse_metrics.png \
  --plot_dpi 800

样本过滤：把“所有 docs + query”的总 token 数先算出来，> --max_L（默认 2050）就直接丢弃，并统计丢弃/保留数量
保证 recompute 和 reuse 的总 token 一致：不再用 tokenizer(full_text) 做 baseline，而是 doc-by-doc tokenize + query tokenize，再把 token ids 拼接成同一条输入，这样两种情况用的 token 序列严格一致
注意力提取不爆显存：用“两步 cache forward”只取 最后一层、最后一个 prompt token 的 attention 向量（q_len=1），并对 heads 做平均
输出：JSONL（每样本的 JSD / Pearson / entropy）+ 3 张图（JSD 直方图、Pearson 直方图、entropy 对比直方图）

- 补充 --doc_mode all：使用样本 context 里的全部文档（含干扰文档）
--doc_mode gold_only：只使用 supporting_facts 覆盖到的文档（不含干扰文档）

- 补充--control_mode recompute_self \或者--control_mode none \来控制对照还是实验组





PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python length_sweep_attention_fixed.py \
  --model_a_path /home/ztl/jwt/models/Qwen2-7B-Instruct \
  --dataset_name hotpotqa \
  --dataset_config distractor \
  --split "train[:30%]" \
  --max_samples 300 \
  --dtype bfloat16 \
  --dump_path true \
  --output_plot kvreuse_metrics.png \
  --max_memory_a "0:10GiB,1:10GiB,2:10GiB,3:10GiB" \
  --use_all_visible_gpus_a true \
  --reserve_gpu0_a 2 \
  --max_L 2050 \
  --plot_dpi 800 \
  --dataset_path /home/ztl/jwt/datasets/hotpotqa/hotpot_qa/distractor \
  --doc_mode gold_only


---取了所有文本，不知道放不放的下---
  PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python length_sweep_attention_fixed.py \
  --model_a_path /home/ztl/jwt/models/Qwen2-7B-Instruct \
  --dataset_name hotpotqa \
  --dataset_config distractor \
  --split "train[:30%]" \
  --max_samples 10 \
  --dtype bfloat16 \
  --dump_path true \
  --output_plot kvreuse_metrics.png \
  --max_memory_a "0:10GiB,1:10GiB,2:10GiB,3:10GiB" \
  --use_all_visible_gpus_a true \
  --reserve_gpu0_a 2 \
  --max_L 2050 \
  --plot_dpi 800 \
  --dataset_path /home/ztl/jwt/datasets/hotpotqa/hotpot_qa/distractor \
  --doc_mode all
