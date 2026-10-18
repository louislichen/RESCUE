import torch
import torch.nn as nn
from shared.modeling_llama_kv_target import LlamaForCausalLM as KVLlamaForCausalLM
from classic.modeling_llama_kv_draft import LlamaForCausalLM as KVLlamaForCausalLM_retrieval
from classic.utils_classic import *
from shared.kv_cache import initialize_past_key_values
from transformers import AutoTokenizer
from shared.opt_tree import Tree
from termcolor import colored
from datetime import datetime
from typing import List, Tuple
import time

from classic.attn_bridge import (
    maybe_apply_attn_bridge,
    maybe_collect_bridge_pairs,
    prepare_prefill_value_bridge,
    reset_bridge_runtime,
)


class SPModel(nn.Module):
    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            draft_model,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self.draft_model = draft_model
        self.draft_stable_kv=None

        self.full_draft_kv=None
        self.evicted = 0
        self.bridge_collector = None
        self.attn_bridge_mapper = None
        self.attn_bridge_config = {}
        reset_bridge_runtime(self)

    def get_tokenizer(self):
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            draft_model_path=None,
            **kwargs,
    ):
        base_model = KVLlamaForCausalLM.from_pretrained(
            base_model_path, **kwargs
        )

        draft_model = KVLlamaForCausalLM_retrieval.from_pretrained(
            draft_model_path, **kwargs
        )

        model = cls(
            base_model,
            base_model_path,
            draft_model
        )

        return model

    @torch.no_grad()
    def ar_generate(
            self,
            input_ids: torch.LongTensor,
            max_new_tokens: int = 256,
            eos_token_id: int = None,
            verbose: bool = False,
            return_metrics: bool = False,
    ):
        """
        Naive autoregressive generation using KV‐cache.

        If verbose=True, prints each newly generated token in green (with proper spacing).
        If verbose=False, returns the full generated string at the end (also properly spaced).

        Args:
            input_ids (torch.LongTensor): shape (1, seq_len), the prompt tokens.
            max_new_tokens (int): maximum number of tokens to append.
            eos_token_id (int, optional): if provided, stop once this token is generated.
            verbose (bool): if True, print each token in green as it's generated.

        Returns:
            If verbose=False: a single string containing prompt + all generated tokens.
            If verbose=True: None (tokens are printed directly with coloring/spacing).
        """
        # Ensure batch size == 1
        assert input_ids.ndim == 2 and input_ids.shape[0] == 1, "batch size >1 is not supported"
        input_len = input_ids.shape[1]
        self.full_cache_budget = input_len + max_new_tokens + 100

        # Build an empty KV cache structure for base_model
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(self.base_model, self.full_cache_budget)
        self.past_key_values = past_key_values
        self.past_key_values_data = past_key_values_data
        self.current_length_data = current_length_data

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # “working” tensor of generated IDs; start with the prompt
        generated = input_ids
        top1_margin_list = []

        # If not verbose, accumulate into this buffer to return later.
        # We decode the *prompt* first.
        full_output = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        for step in range(max_new_tokens):
            if step == 0:
                # First forward: feed the entire prompt, request KV‐cache from the full LM wrapper
                outputs = self.base_model(
                    input_ids=generated,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_kv=True,
                    init=True
                )
            else:
                # On subsequent steps, only feed the last token + past_key_values
                outputs = self.base_model(
                    input_ids=next_token,           # shape (1,1)
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_kv=True
                )

            # Retrieve logits & new KV cache
            logits = outputs.logits

            top_two = torch.topk(logits[:, -1, :].float(), k=2, dim=-1).values
            top1_margin_list.append(float((top_two[:, 0] - top_two[:, 1]).item()))

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)

            generated = torch.cat([generated, next_token], dim=1)

            token_str = self.tokenizer.decode(
                next_token.squeeze(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # If the decoded token does not start with a space, prefix one:
            if token_str and not token_str.startswith(" "):
                spaced_token = " " + token_str
            else:
                spaced_token = token_str

            if verbose:
                # Print in green, flush immediately, no newline
                print(colored(spaced_token, 'green'), end="", flush=True)
            else:
                # Accumulate into buffer (which already contains the prompt)
                full_output += spaced_token

            # Stop early if we hit EOS
            if next_token.item() == eos_token_id:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time
        total_generated = generated.shape[1] - input_len

        if return_metrics:
            return {
                "total_generated": total_generated,
                "inference_time": inference_time,
                "tokens_per_sec": (
                    total_generated / inference_time if inference_time > 0 else 0.0
                ),
                "output_token_ids": generated[0, input_len:].detach().cpu().tolist(),
                "top1_margin_list": top1_margin_list,
            }
        if not verbose:
            return full_output
        else:
            # Print a final newline for cleanliness
            print()
            return

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            tree_attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            nodes=None,
            threshold=None,
            max_depth=None,
            logits_processor=None,
            retrieve_attn_scores=False
    ):
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tree_attention_mask=tree_attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_attentions=True,
                init=init,
                target_use_flash_prefill = self.target_use_flash_prefill,
                target_use_hybrid_tree_attn = self.target_use_hybrid_tree_attn,
                retrieve_attn_scores=retrieve_attn_scores
            )
            
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()
            
            if self.use_retrieval_cache:
                if retrieve_attn_scores:
                    self.attn_scores = outputs.attentions[-1] # already returns last layer attention only
            
        if init:
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

            input_ids, position_ids, tree_attention_mask,parent=self.draft(input_ids,nodes,threshold,max_depth)

            return input_ids,position_ids,tree_attention_mask,token,parent, outputs
        else:
            return outputs, orig, hidden_states

    def process_tree_mask(self, tree_attention_mask, init_len):
        attention_mask=torch.full((tree_attention_mask.size(0), init_len), 0, device=tree_attention_mask.device)
        tree_mask = torch.where(tree_attention_mask == 0, torch.finfo(torch.float32).min, 0)
        attention_mask=torch.cat([attention_mask,tree_mask],dim=-1)
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    @torch.no_grad()
    def draft(self,input_ids,nodes,threshold,max_depth):
        len_posi = input_ids.shape[1]-1
        ###### Initial Forward to generate top_k branches ######
        if hasattr(self, "draft_stable_kv") and self.draft_stable_kv is not None:
            if self.use_retrieval_cache:
                full_kv_len = self.total_seq_len
                draft_outputs = self.draft_model.model(
                    input_ids=input_ids[:, full_kv_len:].to(self.draft_model.model.embed_tokens.weight.device),
                    past_key_values=self.draft_stable_kv,
                    return_kv=True,
                    draft_use_flash_prefill = self.draft_use_flash_prefill
                )
            else:
                kv_len = self.draft_stable_kv[0][0].shape[2]
                draft_outputs = self.draft_model.model(
                    input_ids=input_ids[:, kv_len:].to(self.draft_model.model.embed_tokens.weight.device),
                    past_key_values=self.draft_stable_kv,
                    return_kv=True,
                    draft_use_flash_prefill = self.draft_use_flash_prefill
                )
        ###### Prefill ######
        else:
            bridge_value_overrides = prepare_prefill_value_bridge(
                self, int(input_ids.shape[1])
            )
            prefill_start = prefill_end = None
            draft_device = self.draft_model.model.embed_tokens.weight.device
            if draft_device.type == "cuda":
                prefill_start = torch.cuda.Event(enable_timing=True)
                prefill_end = torch.cuda.Event(enable_timing=True)
                draft_stream = torch.cuda.current_stream(draft_device)
                prefill_start.record(draft_stream)
            draft_outputs = self.draft_model.model(
                input_ids=input_ids.to(self.draft_model.model.embed_tokens.weight.device),
                return_kv=True,
                init=True,
                draft_use_flash_prefill=self.draft_use_flash_prefill,
                bridge_value_overrides=bridge_value_overrides,
            )
            if prefill_end is not None:
                prefill_end.record(draft_stream)
                self.bridge_runtime["draft_prefill_events"].append(
                    (prefill_start, prefill_end)
                )

        if self.use_retrieval_cache:
            newly_appended_len = input_ids.shape[-1] - self.total_seq_len
            self.update_full_draft_cache(draft_outputs[1], tokens_appended=newly_appended_len)
            self.draft_stable_kv = self.update_working_cache_retrieval_main(top_k_chunks=self.retrieve_top_k)
            
            if self.retrieval_verbose:
                self.print_retrieved_chunks()
        else:
            self.draft_stable_kv=draft_outputs[1]

        past_key_values=self.draft_stable_kv

        init_len = past_key_values[0][0].size(2)
        target_model_pos_diff = len_posi - (init_len - 1) 

        last_hidden=draft_outputs[0][:,-1]
        last_headout = self.draft_model.lm_head(last_hidden)
        # Root distribution used to grow this draft tree.  The profiling
        # runner compares it with the target model's root distribution.
        self.last_draft_root_logits = last_headout.detach()

        tree = Tree(nodes, last_hidden.device, threshold, max_depth)

        logits = last_headout.unsqueeze(0)

        step = 0
        while True:
            tree_output = tree.update(
                torch.softmax(logits.to(last_hidden.device), dim=-1, dtype=torch.float32))

            input_ids = tree_output["input_ids"].unsqueeze(0)

            if self.use_retrieval_cache:
                position_ids = tree_output["position_ids"] + init_len-1
            else:
                position_ids = tree_output["position_ids"] + len_posi

            if tree_output["is_final"]:
                break
            tree_attention_mask_with_kv=self.process_tree_mask(tree_output["attention_mask"], init_len)

            draft_outputs = self.draft_model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                tree_attention_mask=tree_attention_mask_with_kv,
                return_kv=True,
                draft_use_flash_prefill = self.draft_use_flash_prefill
            )

            past_key_values=draft_outputs[1]
            last_hidden = draft_outputs[0]
            last_headout = self.draft_model.lm_head(last_hidden)
            logits = last_headout

            step += 1

        if self.use_retrieval_cache:
            position_ids += target_model_pos_diff
        
        return input_ids, position_ids, tree_output["attention_mask"], tree_output["parent_last"]

    @torch.no_grad()
    def spgenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=256,
            nodes=50,
            threshold=0.7,
            max_depth=10,
            output_result_line=False,
            verbose = False,
            retrieval_verbose=False,
            use_specextend=False,
            retrieval_chunk_size=32,
            retrieve_top_k=32,
            retrieve_every_n_steps=8,
            retrieval_policy="attention",
            retrieval_seed=0,
            retrieval_num_queries=1,
            retrieval_query_aggregation="mean",
            retrieval_decay_lambda=0.7,
            dynamic_cache_controller="none",
            dynamic_cache_alpha=0.25,
            dynamic_cache_accept_low=1.0,
            dynamic_cache_accept_high=3.0,
            dynamic_cache_kl_low=0.2,
            dynamic_cache_kl_high=2.5,
            dynamic_cache_patience=2,
            force_flash_prefill=False,
            force_hybrid_tree_attn=False,
            exact_target_attention=False,
            return_output_ids=False,
    ):   
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_len = input_ids.shape[1]
        reset_bridge_runtime(self)
        
        self.use_retrieval_cache = use_specextend
        self.target_use_flash_prefill = (
            (use_specextend or force_flash_prefill) and not exact_target_attention
        )
        self.target_use_hybrid_tree_attn = (
            (use_specextend or force_hybrid_tree_attn) and not exact_target_attention
        )
        self.draft_use_flash_prefill = use_specextend or force_flash_prefill
        
        self.retrieval_chunk_size = retrieval_chunk_size
        self.retrieve_top_k = retrieve_top_k
        self.retrieval_verbose = retrieval_verbose
        self.retrieve_every_n_steps = retrieve_every_n_steps
        if retrieval_policy not in {"attention", "recent", "random"}:
            raise ValueError(f"Unknown retrieval policy: {retrieval_policy}")
        self.retrieval_policy = retrieval_policy
        self.retrieval_seed = retrieval_seed
        if retrieval_num_queries < 0:
            raise ValueError("retrieval_num_queries must be nonnegative; 0 means all accepted tokens")
        if retrieval_query_aggregation not in {"mean", "decay", "max"}:
            raise ValueError("unknown retrieval_query_aggregation")
        self.retrieval_num_queries = retrieval_num_queries
        self.retrieval_query_aggregation = retrieval_query_aggregation
        self.retrieval_decay_lambda = retrieval_decay_lambda
        if dynamic_cache_controller not in {"none", "accept", "kl", "accept_kl"}:
            raise ValueError("unknown dynamic_cache_controller")
        self.dynamic_cache_controller = dynamic_cache_controller
        self.dynamic_cache_alpha = float(dynamic_cache_alpha)
        self.dynamic_cache_accept_low = float(dynamic_cache_accept_low)
        self.dynamic_cache_accept_high = float(dynamic_cache_accept_high)
        self.dynamic_cache_kl_low = float(dynamic_cache_kl_low)
        self.dynamic_cache_kl_high = float(dynamic_cache_kl_high)
        self.dynamic_cache_patience = int(dynamic_cache_patience)
        self.dynamic_cache_ema_accept = None
        self.dynamic_cache_ema_kl = None
        self.dynamic_cache_state = int(retrieve_top_k)
        self.dynamic_cache_pending_state = None
        self.dynamic_cache_pending_count = 0
        self.last_root_target_draft_kl = None
        self.num_chunks_old = 0
        self.retrieval_condition = False
        
        self.attn_scores = None # attention scores returned from the forward pass
        self.attn_scores_final = None # final attention scores used for retrieval (includes newly accepted tokens)

        self.timestep = 0
        
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        self.full_cache_budget = input_len + max_new_tokens + 100

        # initialize caches
        if self.use_retrieval_cache:
            self.init_caches()
        else:
            self.draft_stable_kv = None
            self.full_draft_kv = None
            self.evicted = 0

        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(self.base_model, self.full_cache_budget)
        self.past_key_values = past_key_values
        self.past_key_values_data = past_key_values_data
        self.current_length_data = current_length_data
    
        start_time = datetime.now()
        
        # Prefill target model and draft model + initial draft
        draft_input_ids,draft_position_ids,tree_attention_mask,last_token,parent, outputs = self(
            input_ids=input_ids, past_key_values=past_key_values,  output_orig=True, 
            nodes=nodes, threshold=threshold, max_depth=max_depth, logits_processor=logits_processor
        )

        draft_input_ids=torch.cat([last_token.to(draft_input_ids.device),draft_input_ids],dim=-1)
        draft_position_ids=torch.cat([torch.tensor([draft_position_ids[0]-1],device=draft_position_ids.device), draft_position_ids],dim=-1)
        tree_attention_mask=torch.cat([torch.zeros(1,tree_attention_mask.size(1),dtype=tree_attention_mask.dtype,device=tree_attention_mask.device),tree_attention_mask],dim=0)
        tree_attention_mask = torch.cat([torch.ones(tree_attention_mask.size(0), 1,dtype=tree_attention_mask.dtype,device=tree_attention_mask.device), tree_attention_mask],
                                        dim=1)

        new_token = 0
        total_tokens_list = []
        accept_length_list = []
        while True:
            assert past_key_values[0][0].shape[2]==draft_position_ids[0]
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_input_ids,
                past_key_values,
                draft_position_ids,
                tree_attention_mask
            )

            old_len = input_ids.shape[1]

            input_ids, best_candidate, accept_length, draft_input_ids, draft_position_ids, tree_attention_mask, parent=verify(input_ids,
                                                                      logits,
                                                                      draft_input_ids,
                                                                      draft_position_ids,
                                                                      tree_attention_mask,
                                                                      past_key_values_data,
                                                                      current_length_data,
                                                                      parent,
                                                                      self,
                                                                      nodes,
                                                                      threshold,
                                                                      max_depth,
                                                                      logits_processor)
            
            accept_length_list.append(accept_length.item() if isinstance(accept_length, torch.Tensor) else int(accept_length))

            generated_tokens_list = print_newly_accepted_tokens(old_len, input_ids,
                                                        self.tokenizer, verbose=verbose)
            new_token+=accept_length+1
            total_tokens_list.extend(generated_tokens_list)

            self.timestep += 1

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break

        # Score a fixed requested prefix.  A speculative round can overshoot
        # max_new_tokens by accepting several tokens at once; comparing the raw
        # tails would falsely report exactness failures and bias throughput.
        generated_ids = input_ids[0, input_len:].detach()
        raw_total_generated = int(generated_ids.numel())
        scored_total_generated = min(raw_total_generated, int(max_new_tokens))

        # Calculate eval metrics
        avg_accept_length = round(sum(accept_length_list)/len(accept_length_list), 3)
        inference_time = (datetime.now() - start_time).total_seconds()

        total_generated = scored_total_generated
        tokens_per_sec = round(total_generated/inference_time,2)

        if output_result_line:
            print(colored(
                f"\nGenerated {total_generated} tokens in {inference_time:.2f}s. "
                f"\nToken/sec: {tokens_per_sec}"
                f"\nAverage acceptance length: {avg_accept_length:.3f}",
                'cyan'
            ))

        results = {
            'total_generated': total_generated,
            'raw_total_generated': raw_total_generated,
            'generation_overshoot': raw_total_generated - scored_total_generated,
            'inference_time': inference_time,
            'accept_length_list': accept_length_list,
            'tokens_per_sec': tokens_per_sec,
            'avg_accept_length': avg_accept_length
        }

        if return_output_ids:
            results['output_token_ids'] = generated_ids[:scored_total_generated].cpu().tolist()

        return results

    def update_dynamic_cache_budget(self, accept_length, root_kl):
        """Update the requested retrieval budget for the next draft round."""
        if self.dynamic_cache_controller == "none":
            return
        alpha = self.dynamic_cache_alpha
        accept = float(accept_length.item() if hasattr(accept_length, "item") else accept_length)
        self.dynamic_cache_ema_accept = (
            accept if self.dynamic_cache_ema_accept is None
            else (1.0 - alpha) * self.dynamic_cache_ema_accept + alpha * accept
        )
        if root_kl is not None:
            root_kl = float(root_kl.item() if hasattr(root_kl, "item") else root_kl)
            self.dynamic_cache_ema_kl = (
                root_kl if self.dynamic_cache_ema_kl is None
                else (1.0 - alpha) * self.dynamic_cache_ema_kl + alpha * root_kl
            )

        use_accept = self.dynamic_cache_controller in {"accept", "accept_kl"}
        use_kl = self.dynamic_cache_controller in {"kl", "accept_kl"}
        accept_hard = use_accept and self.dynamic_cache_ema_accept < self.dynamic_cache_accept_low
        accept_easy = (not use_accept) or self.dynamic_cache_ema_accept > self.dynamic_cache_accept_high
        kl_hard = use_kl and self.dynamic_cache_ema_kl is not None and self.dynamic_cache_ema_kl > self.dynamic_cache_kl_high
        kl_easy = (not use_kl) or (
            self.dynamic_cache_ema_kl is not None and self.dynamic_cache_ema_kl < self.dynamic_cache_kl_low
        )
        if accept_hard or kl_hard:
            candidate = 64
        elif accept_easy and kl_easy:
            candidate = 16
        else:
            candidate = 32

        if candidate == self.dynamic_cache_state:
            self.dynamic_cache_pending_state = None
            self.dynamic_cache_pending_count = 0
            return
        if candidate == self.dynamic_cache_pending_state:
            self.dynamic_cache_pending_count += 1
        else:
            self.dynamic_cache_pending_state = candidate
            self.dynamic_cache_pending_count = 1
        if self.dynamic_cache_pending_count >= self.dynamic_cache_patience:
            self.dynamic_cache_state = candidate
            self.retrieve_top_k = candidate
            self.dynamic_cache_pending_state = None
            self.dynamic_cache_pending_count = 0

    def init_caches(self):
        """
        Preallocate both the full cache and the working cache.
        Both are allocated with a budget dimension of full_cache_budget.
        (We assume that full_cache_budget and working_cache_budget are provided;
        here we assume working_cache_budget == sink_size + recent_size.)
        Initially both caches are empty (filled with zeros), and total_seq_len is 0.
        """
        # Get values from the model config.
        num_hidden_layers = self.draft_model.config.num_hidden_layers
        num_heads = self.draft_model.config.num_attention_heads
        head_dim = self.draft_model.config.hidden_size // num_heads

        # full cache: shape: [layers, batch_size, num_heads, full_cache_budget, head_dim]
        self.full_draft_kv = []
        for _ in range(num_hidden_layers):
            full_K = torch.zeros(
                [1, num_heads, self.full_cache_budget, head_dim],
                dtype=torch.float16,
                device=self.draft_model.device
            )
            full_V = torch.zeros(
                [1, num_heads, self.full_cache_budget, head_dim],
                dtype=torch.float16,
                device=self.draft_model.device
            )
            self.full_draft_kv.append((full_K, full_V))

        self.total_seq_len = 0
        self.seq_len_total_old = 0
        self.evicted = 0
        self.draft_stable_kv = None
        self.chunks = None
        self.cached_chunks = None
        # Do not let the previous request's selected chunk IDs leak into the
        # next request.  Upstream only checked hasattr(), so stale selections
        # were reused for the first draft step of later samples.
        self.selected_chunks = None
        self.draft_model.model.past_key_position_ids = None
        
        self.recent_start = 0
        self.recent_end = 0

    def update_full_draft_cache(self, new_kv: List[Tuple[torch.Tensor, torch.Tensor]], tokens_appended: int):
        """
        Update the full draft KV cache with the new tokens.
        new_kv is the returned KV from the forward pass (a working-cache view).
        tokens_appended is the number of new tokens processed in this forward pass.
        
        The full cache is preallocated with size self.full_cache_budget, and
        self.total_seq_len tracks the current number of tokens stored.
        This function copies the last tokens_appended tokens from new_kv (from the working view)
        into the full cache.
        """
        # Check that we don't exceed the allocated budget.
        if self.total_seq_len + tokens_appended > self.full_cache_budget:
            raise RuntimeError(
                f"Full cache budget exceeded: total_seq_len {self.total_seq_len} + new {tokens_appended} > {self.full_cache_budget}"
            )

        # Precompute destination slice indices.
        dest_start = self.total_seq_len
        dest_end = dest_start + tokens_appended
        device = self.draft_model.device

        # For each layer in the new KV, copy the last tokens_appended tokens into the full cache.
        for i, (new_K, new_V) in enumerate(new_kv):
            full_K, full_V = self.full_draft_kv[i]
            # Ensure new_K and new_V are on the correct device.
            new_K = new_K.to(device, non_blocking=True)
            new_V = new_V.to(device, non_blocking=True)
            # Copy the last tokens_appended tokens from new_K/new_V into the full cache.
            full_K[:, :, dest_start:dest_end, :].copy_(new_K[:, :, -tokens_appended:, :])
            full_V[:, :, dest_start:dest_end, :].copy_(new_V[:, :, -tokens_appended:, :])
        
        # Update the total sequence length.
        self.total_seq_len = dest_end
        
    def prepare_chunks(self):
        """
        Called once (right after prefill) to split the full cache (of length self.total_seq_len)
        into consecutive chunks of fixed size (self.retrieval_chunk_size). Each chunk is represented as a tuple:
        (chunk_idx, start, end) where end - start <= self.retrieval_chunk_size.
        """
        self.chunks = []
        current_start = 0
        chunk_idx = 0
        while current_start < self.total_seq_len:
            end_pos = min(current_start + self.retrieval_chunk_size, self.total_seq_len)
            self.chunks.append((chunk_idx, current_start, end_pos))
            chunk_idx += 1
            current_start = end_pos
        # Save the current full length so that later we know how many new tokens were appended.
        self.seq_len_total_old = self.total_seq_len
        self.num_chunks = len(self.chunks)
        self.num_chunks_old = self.num_chunks

    def update_chunks(self):
        """
        Called after new tokens have been appended to the full KV cache.
        self.total_seq_len has been updated externally (by update_full_draft_cache).
        This function updates self.chunks to reflect the new total length.
        
        It does so by:
        1) Filling the last chunk (if not already full) with some of the new tokens.
        2) Creating new chunk(s) (each of size self.retrieval_chunk_size, except possibly the last one)
            for any remaining new tokens.
        """
        # Calculate how many new tokens were appended.
        new_tokens = self.total_seq_len - self.seq_len_total_old
        if new_tokens <= 0:
            return  # No new tokens; nothing to do.

        # If there are no chunks yet, create them from scratch.
        if not hasattr(self, "chunks") or self.chunks is None or len(self.chunks) == 0:
            self.prepare_chunks()
            return

        # Get the last chunk's info.
        last_chunk_idx, last_start, last_end = self.chunks[-1]
        last_chunk_size = last_end - last_start
        remaining_new_tokens = new_tokens

        # 1) If the last chunk is not full, fill it up as much as possible.
        capacity = self.retrieval_chunk_size - last_chunk_size
        if capacity > 0:
            tokens_to_add = min(capacity, remaining_new_tokens)
            # Update the last chunk's end index.
            self.chunks[-1] = (last_chunk_idx, last_start, last_end + tokens_to_add)
            remaining_new_tokens -= tokens_to_add
    
        # 2) For any remaining tokens, create new chunks.
        current_start = self.total_seq_len - remaining_new_tokens
        while remaining_new_tokens > 0:
            tokens_in_chunk = min(self.retrieval_chunk_size, remaining_new_tokens)
            new_chunk = (self.chunks[-1][0] + 1, current_start, current_start + tokens_in_chunk)
            self.chunks.append(new_chunk)
            current_start += tokens_in_chunk
            remaining_new_tokens -= tokens_in_chunk

        # Update the stored old full length.
        self.seq_len_total_old = self.total_seq_len
        self.num_chunks = len(self.chunks)

        if self.num_chunks > self.num_chunks_old:
            self.num_chunks_old = self.num_chunks
            return True # new chunk was added => update working cache
        return False

    def update_working_cache_retrieval(self, top_k_chunks: int = 15,
                                       do_retrieval=False,
                                       is_updated_chunks=False) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # initial cache: use recent chunks (we don't have attn scores yet)
        if not hasattr(self, "selected_chunks") or self.selected_chunks is None:
            num_init = min(self.retrieve_top_k, len(self.chunks))
            if self.retrieval_policy == "random":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.retrieval_seed)
                other_count = max(0, num_init - 1)
                indices = torch.randperm(
                    max(0, len(self.chunks) - 1), generator=generator
                )[:other_count].tolist()
                indices.append(len(self.chunks) - 1)
                self.selected_chunks = sorted(
                    [self.chunks[i] for i in indices], key=lambda x: x[0]
                )
            else:
                self.selected_chunks = self.chunks[-num_init:]
        
        # Only retrieve top-k upon retrieval condition
        if do_retrieval:
            n = len(self.chunks)
            if n == 0:
                raise ValueError("No chunks available for retrieval.")

            k = min(top_k_chunks, n)

            if self.retrieval_policy == "recent":
                selected_chunks = self.chunks[-k:]
            elif self.retrieval_policy == "random":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.retrieval_seed + self.timestep)
                other_count = max(0, k - 1)
                indices = torch.randperm(max(0, n - 1), generator=generator)[:other_count].tolist()
                indices.append(n - 1)
                selected_chunks = [self.chunks[i] for i in indices]
            else:
                attn = self.attn_scores_final

                chunks_tensor = torch.tensor(
                    [[start, end] for (_, start, end) in self.chunks],
                    dtype=torch.long,
                    device=attn.device
                )

                starts = chunks_tensor[:, 0]  # shape: [num_chunks]
                ends = chunks_tensor[:, 1]    # shape: [num_chunks]

                # Compute cumulative sum of attn scores for fast range-sum computation
                cum_attn = torch.cumsum(attn, dim=0) # shape: [L]
    
                # For each chunk, the sum is cum_attn[ends-1] - (cum_attn[starts-1] if start>0 else 0)
                lower = torch.where(starts > 0, cum_attn[starts - 1], torch.zeros_like(starts, dtype=attn.dtype))
                ends_minus_one = torch.clamp(ends - 1, max=cum_attn.size(0) - 1)
                chunk_sums = cum_attn[ends_minus_one] - lower

                # Compute lengths and then means (cast lengths to float)
                lengths = (ends - starts).float()
                chunk_means = chunk_sums / lengths  # shape: [num_chunks]
            
                other_count = max(0, k - 1)
                topk = torch.topk(chunk_means[:-1], k=other_count)
                selected_indices = topk.indices  # indices into the list of chunks
                selected_chunks = [self.chunks[i] for i in selected_indices.tolist()]
                selected_chunks.append(self.chunks[-1])

            selected_chunks.sort(key=lambda x: x[0])
            self.selected_chunks = selected_chunks

            # reset retrieval condition and attn scores
            self.retrieval_condition = False
            self.attn_scores = None
            self.attn_scores_final = None

        # if new chunk is added, automatically update
        if is_updated_chunks:
            # grab the newly created chunk
            new_chunk = self.chunks[-1]  # (chunk_id, start, end)
            new_chunk_id = new_chunk[0]
            existing_ids = {cid for cid, _, _ in self.selected_chunks}
            # only append it if it’s not already in the selected set
            if new_chunk_id not in existing_ids:
                self.selected_chunks.append(new_chunk)
            # Keep a strict chunk budget between retrieval rounds.  The newest
            # (causal frontier) chunk is mandatory; evict the oldest retained
            # chunk if adding it would otherwise create a K+1 cache.
            if len(self.selected_chunks) > top_k_chunks:
                self.selected_chunks = sorted(self.selected_chunks, key=lambda x: x[0])
                self.selected_chunks = self.selected_chunks[-top_k_chunks:]
                
        # update last selected chunk
        if self.selected_chunks[-1][0] == self.chunks[-1][0]:
            chunk_id, start, _ = self.selected_chunks[-1]
            new_end = self.chunks[-1][2]
            self.selected_chunks[-1] = (chunk_id, start, new_end)
                
        all_indices = []
        for (_, start, end) in self.selected_chunks:
            all_indices.extend(range(start, end))
        if len(all_indices) == 0:
            raise ValueError("No tokens retrieved from the full cache. Check your chunk settings and attn_scores_final.")
        
        retrieved_indices = torch.tensor(all_indices, dtype=torch.long)
        retrieved_indices = torch.unique(retrieved_indices, sorted=True).to(self.draft_model.device)

        if retrieved_indices.numel() == 0:
            raise ValueError("No tokens retrieved from the full cache. Check your chunk settings and attn_scores_final.")

        # Build working cache by advanced indexing into full cache for each layer
        working_kv = []
        for (full_K, full_V) in self.full_draft_kv:
            working_K_layer = full_K.index_select(dim=2, index=retrieved_indices)
            working_V_layer = full_V.index_select(dim=2, index=retrieved_indices)
            working_kv.append((working_K_layer, working_V_layer))
        maybe_collect_bridge_pairs(self)
        working_kv = maybe_apply_attn_bridge(self, retrieved_indices, working_kv)
        self.draft_stable_kv = working_kv

        # Update evicted count
        self.evicted = self.total_seq_len - retrieved_indices.numel()

        # Update past_key_position_ids
        past_ids = self.draft_model.model.past_key_position_ids  # shape [1, current_length]
        current_length = past_ids.shape[1]
        target_length = retrieved_indices.numel()
        if current_length < target_length:
            extra_ids = torch.arange(current_length, target_length, device=past_ids.device).unsqueeze(0)
            new_past_ids = torch.cat([past_ids, extra_ids], dim=1)
        else:
            new_past_ids = past_ids[:, :target_length]
        self.draft_model.model.past_key_position_ids = new_past_ids

        if self.retrieval_verbose:
            if self.retrieval_condition:
                self.print_retrieved_chunks(order="id")
        return working_kv


    def update_working_cache_retrieval_main(self, top_k_chunks: int = 15):
        """
        Convenience function that first updates the chunk metadata (if new tokens were appended)
        and then updates the working cache (self.draft_stable_kv) based on retrieval.
        
        It assumes that self.attn_scores_final is already set (e.g., computed from the last accepted query)
        and that self.total_seq_len has been updated by update_full_draft_cache.
        """
        is_updated_chunks = self.update_chunks()

        working_kv = self.update_working_cache_retrieval(top_k_chunks=top_k_chunks,
                                                            do_retrieval=self.retrieval_condition,
                                                            is_updated_chunks=is_updated_chunks) 
        return working_kv
        
    def print_retrieved_chunks(self, order="id"):
        if order == "score":
            chunks_list = self.selected_chunks
            msg = "\nRetrieved chunk IDs (descending attn score): "
        elif order == "id":
            chunks_list = sorted(self.selected_chunks, key=lambda x: x[0])
            msg = "\nRetrieved chunk IDs: "
        else:
            print(colored(f"Unknown 'order' option: {order}. Choose 'score' or 'id'.", 'red'))
            return

        chunk_ids_str = ", ".join(str(chunk[0]) for chunk in chunks_list)

        print(colored(msg + chunk_ids_str + '\n', 'yellow'))
