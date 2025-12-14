--用来测试基础对不对---

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python length_sweep_attention.py \
  --model_a_path /home/ztl/jwt/models/Qwen2-7B-Instruct \
  --model_b_path /home/ztl/jwt/models/Qwen2-1_5B_Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --dataset_path /home/ztl/jwt/datasets/wikitext/wikitext-2-raw-v1 \
  --split "train[:10%]" \
  --max_samples 1 \
  --dtype bfloat16 \
  --attention_precision 6 \
  --dump_path true \
  --output_plot length_sweep_corr.png \
  --max_memory_a "0:7GiB,1:10GiB,2:10GiB,3:10GiB" \
  --use_all_visible_gpus_a true \
  --reserve_gpu0_a 2 \
  --sweep_step 10 \
  --max_L 2050 \
  --plot_dpi 800

--实验代码---
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
python length_sweep_attention.py \
  --model_a_path /home/ztl/jwt/models/Qwen2-7B-Instruct \
  --model_b_path /home/ztl/jwt/models/Qwen2-1_5B_Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --dataset_path /home/ztl/jwt/datasets/wikitext/wikitext-2-raw-v1 \
  --split "train[:10%]" \
  --max_samples 300 \
  --dtype bfloat16 \
  --attention_precision 6 \
  --dump_path true \
  --output_plot length_sweep_corr.png \
  --max_memory_a "0:7GiB,1:10GiB,2:10GiB,3:10GiB" \
  --use_all_visible_gpus_a true \
  --reserve_gpu0_a 2 \
  --sweep_step 10 \
  --max_L 2050 \
  --plot_dpi 800



  --对照组---
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
python length_sweep_attention.py \
  --model_a_path /home/ztl/jwt/models/Qwen2-1_5B_Instruct \
  --model_b_path /home/ztl/jwt/models/Qwen2-1_5B_Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --dataset_path /home/ztl/jwt/datasets/wikitext/wikitext-2-raw-v1 \
  --split "train[:10%]" \
  --max_samples 10 \
  --dtype bfloat16 \
  --attention_precision 6 \
  --dump_path true \
  --output_plot length_sweep_corr.png \
  --max_memory_a "0:7GiB,1:10GiB,2:10GiB,3:10GiB" \
  --use_all_visible_gpus_a true \
  --reserve_gpu0_a 2 \
  --sweep_step 10 \
  --max_L 2050 \
  --plot_dpi 800

--测试组---还是跑不上去
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
python length_sweep_attention.py \
  --model_a_path /home/ztl/jwt/models/Qwen2-7B-Instruct \
  --model_b_path /home/ztl/jwt/models/Qwen2-1_5B_Instruct \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --dataset_path /home/ztl/jwt/datasets/wikitext/wikitext-2-raw-v1 \
  --split "train[:10%]" \
  --max_samples 1 \
  --dtype bfloat16 \
  --attention_precision 6 \
  --dump_path true \
  --output_plot length_sweep_corr.png \
  --max_memory_a "0:7GiB,1:10GiB,2:10GiB,3:10GiB" \
  --use_all_visible_gpus_a true \
  --reserve_gpu0_a 2 \
  --sweep_step 10 \
  --max_L 3000 \
  --plot_dpi 800 

1. 切分到四卡同时工作，因为2048时候的长上下文OOM了
2. 四卡跑2048也OOM，只取最后一层的注意力,优化，优化是成功了，跑到了2048也不炸了，但是又开始回弹了没绷住
3. 增加了一个JS散度
4. 采用取下三角的方式来采集样本点，这样一个2050token只需要做一次prefill取下三角就可以了
5. 5,3对图像做美化

1) --max_memory_a "0:7GiB,1:10GiB,2:10GiB,3:10GiB"
这是在告诉 HuggingFace/Accelerate：
给 model A 做多卡切分时，每张 GPU 最多允许用多少显存来放模型权重/中间模块。
0:7GiB 表示：GPU0 最多让它装到 7GiB 左右
1:10GiB、2:10GiB、3:10GiB 类似
为什么要故意把 GPU0 压低？
你 OOM 的那条栈在 softmax(attn_weights)，这属于 forward 过程中的临时峰值显存（activation/临时buffer），不是单纯“模型权重”占满。
如果 GPU0 上已经塞了太多模型层/权重，它剩余空间就不够给这些峰值用，于是就爆。
把 0 的预算压低，会逼着 device_map 把更多层挪到 GPU1/2/3，让 GPU0 **空出一块“缓冲区”**专门承受 forward 峰值。
简单说：max_memory_a 不是让你“用满显存”，而是“控制切分策略”，让某张卡别太满。
2) --use_all_visible_gpus_a true
这个参数是我新增的，作用是：
自动把 model A 的 max_memory_a 补齐到所有可见 GPU（0..N-1），避免你漏写某张卡导致它完全不参与切分。
你之前遇到的情况是“第四张卡几乎空载”，常见原因之一就是：
你给 max_memory_a 只写了 0,1,2（没写 3），那 accelerate 在规划时就可能把 GPU3 当成“不允许使用”，自然就空着。
开了这个参数后，就算你写的是：
--max_memory_a "0:7GiB,1:10GiB,2:10GiB"
脚本也会自动补成：
{0:"7GiB",1:"10GiB",2:"10GiB",3:"10GiB"}
（假设你 CUDA_VISIBLE_DEVICES=0,1,2,3）
3) --reserve_gpu0_a 2
这也是我新增的，作用是：
在脚本内部自动从 GPU0 的 max_memory 里再扣掉 2GiB，给 forward 的临时峰值留“安全垫”。
它和你手动写 0:7GiB 的区别：
你手动 0:7GiB：你自己决定 GPU0 给多少
reserve_gpu0_a 2：脚本会在你给的基础上再减 2GiB
例如你写 0:9GiB，reserve=2 会变成 0:7GiB
为什么 reserve 只对 GPU0 做？
因为通常输入 embedding、前几层、以及一些关键算子更可能落在第一张卡（尤其是自动切分时），GPU0 更容易成为“峰值撞墙点”。给它留余量，最能减少 OOM。
三者配合起来的直觉
max_memory_a：你给每张卡设“装模型的上限”，让切分更均衡
use_all_visible_gpus_a：确保 GPU3 一定被纳入切分候选
reserve_gpu0_a：让 GPU0 别太满，给 2048 的 attention/softmax 峰值留缓冲
你怎么判断它真的生效了？
你跑起来后，代码会打印类似：
model A max_memory (after auto-fix): {0:'7GiB',1:'10GiB',2:'10GiB',3:'10GiB'}
model A hf_device_map module counts: cuda:0:xxx, cuda:1:yyy, cuda:2:zzz, cuda:3:www
如果 cuda:3 的数量不是 0，就说明第四张卡确实参与了模型切分。


================ SUMMARY ================
Groups evaluated : 1
L=  32 | Pearson n=  1 mean=0.9621189637638045 std=0.0 | Spearman n=  1 mean=0.6363636363636364 std=0.0
L=  64 | Pearson n=  1 mean=0.9405495246538178 std=0.0 | Spearman n=  1 mean=0.7243901601912346 std=0.0
L= 128 | Pearson n=  1 mean=0.8846094463925362 std=0.0 | Spearman n=  1 mean=0.6769193793436928 std=0.0
L= 256 | Pearson n=  1 mean=0.9898287980205288 std=0.0 | Spearman n=  1 mean=0.5976255896744739 std=0.0
L= 384 | Pearson n=  1 mean=0.903473164641217 std=0.0 | Spearman n=  1 mean=0.6928487617831902 std=0.0
L= 512 | Pearson n=  1 mean=0.9618102604136627 std=0.0 | Spearman n=  1 mean=0.8129502534500485 std=0.0
L=1024 | Pearson n=  1 mean=0.7714856579006182 std=0.0 | Spearman n=  1 mean=0.7348870562881188 std=0.0
JSONL saved     : /home/ztl/jwt/Lab/Lab20/length_sweep_Qwen2-7B-Instruct__vs__Qwen2-1_5B_Instruct.jsonl

基本对齐