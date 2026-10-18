import json
import torch
import torch.nn as nn
from transformers import AutoConfig
from shared.modeling_llama_kv_target import LlamaForCausalLM as KVLlamaForCausalLM
from eagle.utils_eagle import *
from shared.kv_cache import initialize_past_key_values
from transformers import AutoTokenizer
import os 
from huggingface_hub import hf_hub_download
from eagle.cnets import Model
from eagle.configs_eagle import EConfig
from huggingface_hub import hf_hub_download
from termcolor import colored
from datetime import datetime

def tensor_size_bytes(tensor):
    return tensor.element_size() * tensor.nelement()

def total_kv_cache_size(past_key_values):
    total = 0
    for layer in past_key_values:
        for kv in layer:
            # Adjust "kv.data" if your KVCache stores its tensor differently.
            total += tensor_size_bytes(kv.data)
    return total

class EaModel(nn.Module):
    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                # self.ea_layer.head=nn.Linear(base_model.lm_head.in_features,base_model.lm_head.out_features,bias=False)
                # self.ea_layer.head.weight=copy.deepcopy(base_model.lm_head.weight)
                # self.ea_layer.head.to(device)
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.tokenizer = self.tokenizer

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            **kwargs,
    ):
        #assert Type=="LLaMA"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
    
        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath
        )
        load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path,
                                         map_location=base_model.device)
        model.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        model.ea_layer.device = model.ea_layer.embed_tokens.weight.device
        model.ea_layer.base_config = model.config

        return model

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
            # Prefill target model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tree_attention_mask=tree_attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_attentions=True,
                init=init,
                target_use_flash_prefill = self.ea_layer.target_use_flash_prefill,
                target_use_hybrid_tree_attn = self.ea_layer.target_use_hybrid_tree_attn,
                retrieve_attn_scores=retrieve_attn_scores
            )
            
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()
            
            if self.ea_layer.use_retrieval_cache:
                if retrieve_attn_scores:
                    self.ea_layer.attn_scores = outputs.attentions[-1] # already returns last layer attention only

        # initial tree draft
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
            # Clone the output hidden states

            input_ids,position_ids,tree_attention_mask,parent = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head, nodes=nodes,threshold=threshold,max_depth=max_depth)
            return input_ids,position_ids,tree_attention_mask,token,parent
        else:

            return outputs, orig, hidden_states

    @torch.no_grad()
    def eagenerate(
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
            verbose=True,
            retrieval_verbose=False,
            use_specextend=False,
            retrieval_chunk_size = 32,
            retrieve_top_k = 64,
            retrieve_every_n_steps = 8,
            retrieval_policy="attention",
            retrieval_seed=0,
            retrieval_num_queries=1,
            retrieval_query_aggregation="mean",
            retrieval_decay_lambda=0.7,
            force_flash_prefill=False,
            force_hybrid_tree_attn=False,
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_len = input_ids.shape[1]

        self.ea_layer.use_retrieval_cache = use_specextend
        self.ea_layer.target_use_flash_prefill = use_specextend or force_flash_prefill
        self.ea_layer.target_use_hybrid_tree_attn = use_specextend or force_hybrid_tree_attn
        self.ea_layer.draft_use_flash_prefill = use_specextend or force_flash_prefill
        
        self.ea_layer.retrieval_chunk_size = retrieval_chunk_size
        self.ea_layer.retrieve_top_k = retrieve_top_k
        self.ea_layer.retrieval_verbose = retrieval_verbose
        self.ea_layer.retrieve_every_n_steps=retrieve_every_n_steps
        if retrieval_policy not in {"attention", "recent", "random"}:
            raise ValueError(f"Unknown retrieval policy: {retrieval_policy}")
        self.ea_layer.retrieval_policy = retrieval_policy
        self.ea_layer.retrieval_seed = retrieval_seed
        if retrieval_num_queries < 0:
            raise ValueError("retrieval_num_queries must be nonnegative; 0 means all accepted tokens")
        if retrieval_query_aggregation not in {"mean", "decay", "max"}:
            raise ValueError("unknown retrieval_query_aggregation")
        self.ea_layer.retrieval_num_queries = retrieval_num_queries
        self.ea_layer.retrieval_query_aggregation = retrieval_query_aggregation
        self.ea_layer.retrieval_decay_lambda = retrieval_decay_lambda
        self.ea_layer.num_chunks_old = 0
        self.ea_layer.retrieval_condition = False
        
        self.ea_layer.attn_scores = None
        self.ea_layer.attn_scores_final = None

        self.ea_layer.timestep = 0

        input_len = input_ids.shape[1]

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()
        
        self.ea_layer.full_cache_budget = input_len + max_new_tokens + 100

        # initialize draft model caches
        if self.ea_layer.use_retrieval_cache:
            self.init_caches()
        else:
            # self.ea_layer.reset_kv()
            self.ea_layer.draft_stable_kv = None
            self.ea_layer.full_draft_kv = None
            self.ea_layer.evicted = 0

        # Initialize target model caches
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(self.base_model, self.ea_layer.full_cache_budget)
        self.past_key_values = past_key_values
        self.past_key_values_data = past_key_values_data
        self.current_length_data = current_length_data

        start_time = datetime.now()
        
        draft_input_ids,draft_position_ids,tree_attention_mask,last_token,parent=self(input_ids, past_key_values=past_key_values, output_orig=True, nodes=nodes, threshold=threshold, max_depth=max_depth,logits_processor=logits_processor)

        draft_input_ids=torch.cat([last_token,draft_input_ids],dim=-1)
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
                tree_attention_mask,
            )
            
            old_len = input_ids.shape[1]
            
            input_ids,best_candidate,accept_length,draft_input_ids,draft_position_ids,tree_attention_mask,parent=verify(input_ids,
                                                                      logits,
                                                                      draft_input_ids,
                                                                      draft_position_ids,
                                                                      hidden_state_new,
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

            self.ea_layer.timestep += 1

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break

        # Calculate eval metrics
        avg_accept_length = round(sum(accept_length_list)/len(accept_length_list), 3)
        inference_time = (datetime.now() - start_time).total_seconds()
        total_generated = new_token
        tokens_per_sec = round(total_generated/inference_time,2)
        
        if output_result_line:
            print(colored(
                f"\nGenerated {total_generated} tokens in {inference_time:.2f}s. "
                f"\nToken/sec: {tokens_per_sec}"
                f"\nAverage acceptance length: {avg_accept_length:.3f}",
                'cyan'
            ))

        results = {
            'avg_accept_length': avg_accept_length,
            'total_generated': total_generated,
            'inference_time': inference_time,
            'tokens_per_sec': tokens_per_sec,
        }

        return results

    def init_caches(self):
        """
        Preallocate both the full cache and the working cache.
        Both are allocated with a budget dimension of full_cache_budget.
        (We assume that full_cache_budget and working_cache_budget are provided;
        here we assume working_cache_budget == sink_size + recent_size.)
        Initially both caches are empty (filled with zeros), and total_seq_len is 0.
        """
        # Get values from the model config.
        num_hidden_layers = self.ea_layer.config.num_hidden_layers

        num_heads = self.ea_layer.config.num_attention_heads
        head_dim = self.ea_layer.config.hidden_size // num_heads
        # num_heads = self.base_model.config.num_attention_heads
        # head_dim = self.base_model.config.hidden_size // num_heads

        # full cache: shape: [layers, batch_size, num_heads, full_cache_budget, head_dim]
        self.ea_layer.full_draft_kv = []
        for layer_idx in range(num_hidden_layers):
            device = self.ea_layer.layers[layer_idx].self_attn.q_proj.weight.device
            full_K = torch.zeros(
                [1, num_heads, self.ea_layer.full_cache_budget, head_dim],
                dtype=torch.float16,
                device=device
            )
            full_V = torch.zeros(
                [1, num_heads, self.ea_layer.full_cache_budget, head_dim],
                dtype=torch.float16,
                device=device
            )
            self.ea_layer.full_draft_kv.append((full_K, full_V))

        self.ea_layer.total_seq_len = 0
        self.ea_layer.seq_len_total_old = 0
        self.ea_layer.evicted = 0
        self.ea_layer.draft_stable_kv = None
        self.ea_layer.chunks = None
        self.ea_layer.cached_chunks = None
        self.ea_layer.selected_chunks = None
        self.ea_layer.past_key_position_ids = None

        self.ea_layer.recent_start = 0
        self.ea_layer.recent_end = 0
