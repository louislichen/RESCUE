import argparse

from ...tools.plot import *
from ...tools.corrs import *
from ...tools.utils import *


def parse_args():
    p = argparse.ArgumentParser("Length sweep attention similarity (Route A + Multi-GPU + JSONL + Plot)")

    p.add_argument("--plot_dpi", type=int, default=300, help="PNG DPI (paper-like default: 300)")

    p.add_argument("--device_map_a", type=str, default="auto")
    p.add_argument("--device_map_b", type=str, default="auto")

    p.add_argument("--max_memory_a", type=str, default="0:10GiB,1:10GiB,2:10GiB,3:10GiB")
    p.add_argument("--max_memory_b", type=str, default="0:10GiB,1:10GiB,2:10GiB,3:10GiB")

    p.add_argument("--use_all_visible_gpus_a", type=str2bool, default=True)
    p.add_argument("--reserve_gpu0_a", type=float, default=2.0)

    p.add_argument("--empty_cache_each_group", type=str2bool, default=True)

    return p.parse_args()

def main():
    
    pearson_by_L: Dict[int, List[float]] = {L: [] for L in L_list}
    spearman_by_L: Dict[int, List[float]] = {L: [] for L in L_list}
    jsd_by_L: Dict[int, List[float]] = {L: [] for L in L_list}

    print(f">>> Route A enabled: each group token length >= {required_L} (required).")

    total = 0
    group_index = 0
    i = 0

    pbar = tqdm(total=args.max_samples, desc="Groups")

    while total < args.max_samples:
        if n is not None and i >= n:
            break

        texts: List[str] = []
        token_len = 0

        while token_len < required_L:
            if n is not None and i >= n:
                break
            ex = ds[i]
            i += 1
            t = ex.get("text")
            if not t or not t.strip():
                continue
            texts.append(t.strip())
            enc_tmp = tokenizer("\n\n".join(texts), return_tensors="pt", truncation=False)
            token_len = int(enc_tmp["input_ids"].shape[-1])

        if token_len < required_L:
            break

        group_text = "\n\n".join(texts)
        enc = tokenizer(group_text, return_tensors="pt", truncation=False)
        input_ids_full = enc["input_ids"][0]
        attn_mask_full = enc["attention_mask"][0] if "attention_mask" in enc else None
        T = int(input_ids_full.shape[0])

        if T < required_L:
            continue

        input_ids_max = input_ids_full[:required_L].unsqueeze(0)
        attn_mask_max = attn_mask_full[:required_L].unsqueeze(0) if attn_mask_full is not None else None

        dev_a = get_input_device(model_a)
        dev_b = get_input_device(model_b)

        input_ids_a = input_ids_max.to(dev_a)
        mask_a = attn_mask_max.to(dev_a) if attn_mask_max is not None else None

        input_ids_b = input_ids_max.to(dev_b)
        mask_b = attn_mask_max.to(dev_b) if attn_mask_max is not None else None

        vecs_a = sampled_attention_rows_last_layer(model_a, input_ids_a, mask_a, L_list, args.attention_precision)
        vecs_b = sampled_attention_rows_last_layer(model_b, input_ids_b, mask_b, L_list, args.attention_precision)

        for L in L_list:
            vec_a = vecs_a.get(L, None)
            vec_b = vecs_b.get(L, None)

            pear = None
            spear = None
            jsd = None

            if vec_a is not None and vec_b is not None and is_finite_list(vec_a) and is_finite_list(vec_b):
                pear = pearson_corr(vec_a, vec_b)
                spear = spearman_corr(vec_a, vec_b)
                jsd = js_divergence(vec_a, vec_b)

                if pear is not None and math.isfinite(pear):
                    pearson_by_L[L].append(float(pear))
                if spear is not None and math.isfinite(spear):
                    spearman_by_L[L].append(float(spear))
                if jsd is not None and math.isfinite(jsd):
                    jsd_by_L[L].append(float(jsd))
            else:
                vec_a = None
                vec_b = None

            if dump_fh is not None:
                obj = {
                    "sample_index": group_index,
                    "L": L,
                    "generated": "",
                    "generated_tokens": [],
                    "last_layer_attention_a": [vec_a] if vec_a is not None else None,
                    "last_layer_attention_b": [vec_b] if vec_b is not None else None,
                    "pearson": None if pear is None or (isinstance(pear, float) and not math.isfinite(pear)) else round(float(pear), 6),
                    "spearman": None if spear is None or (isinstance(spear, float) and not math.isfinite(spear)) else round(float(spear), 6),
                    "jsd": None if jsd is None or (isinstance(jsd, float) and not math.isfinite(jsd)) else round(float(jsd), 6),
                    "group_total_tokens": T,
                    "group_num_segments": len(texts),
                }
                dump_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

        del input_ids_a, mask_a, input_ids_b, mask_b, input_ids_max, attn_mask_max
        torch.cuda.empty_cache()

        group_index += 1
        total += 1
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()
