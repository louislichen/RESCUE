import json
import time
import gradio as gr
import torch
from accelerate import Accelerator
from transformers import LlamaConfig, AutoTokenizer
from application.model_classic import SPModel
from application.modeling_llama import LlamaForCausalLM
from termcolor import colored
import gc

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
ar_model = None
ar_tokenizer = None

sp_model = None
sp_tokenizer = None

accelerator = Accelerator()

# -----------------------------------------------------------------------------
# 1) JSONL loader
# -----------------------------------------------------------------------------
def load_texts_from_jsonl(path: str, max_samples: int = None):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if text is None:
                continue
            texts.append(text)
    return texts

# -----------------------------------------------------------------------------
# 2) Load prompt & (optional) warm up SPModel
# -----------------------------------------------------------------------------
BASE_SP_MODEL  = "lmsys/vicuna-7b-v1.5-16k"
DRAFT_SP_MODEL = "double7/vicuna-68m"

def load_and_warm(dataset: str, length: str) -> str:
    """
    Load first sample from data/{dataset}/{dataset}_{length}.jsonl,
    but if dataset=="pg-19", strip the hyphen when building the filename.
    If SPModel is already loaded, run 3 quick warm-up passes on that text.
    """
    global sp_model, sp_tokenizer

    dataset_lower = dataset.lower()  # e.g. "govreport" or "pg-19"
    # If the folder is "pg-19", we need to remove the hyphen for the filename prefix:
    if dataset_lower == "pg-19":
        prefix = dataset_lower.replace("-", "")  # "pg19"
    else:
        prefix = dataset_lower  # e.g. "govreport"

    path = f"data/{dataset_lower}/{prefix}_{length}.jsonl"
    try:
        texts = load_texts_from_jsonl(path, max_samples=1)
        if not texts:
            return ""
        text = texts[0]
    except FileNotFoundError:
        return ""

    # If SPModel is in memory, warm it up with 3 short runs
    if sp_model:
        encoded = sp_tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(accelerator.device)
        for _ in range(3):
            gen = sp_model.spgenerate(
                input_ids,
                temperature=0.0,
                max_new_tokens=10,
                use_specextend=True,
                retrieval_chunk_size=32,
                retrieve_top_k=32,
                retrieve_every_n_steps=4
            )
            for _ in gen:
                pass

    return text

# -----------------------------------------------------------------------------
# 3) Load AR model on demand
# -----------------------------------------------------------------------------
def load_ar_model() -> str:
    """
    Delete any existing AR model (and free its GPU memory), then load a fresh LlamaForCausalLM.
    """
    global ar_model, ar_tokenizer

    # If an AR model exists, move it to CPU, delete it, and clear cache
    if ar_model is not None:
        try:
            ar_model.cpu()
        except:
            pass
        try:
            del ar_model
            del ar_tokenizer
        except:
            pass

        torch.cuda.empty_cache()
        gc.collect()

        ar_model = None
        ar_tokenizer = None

    # Load a fresh autoregressive Llama model onto GPU
    base_model_path = "lmsys/vicuna-7b-v1.5-16k"
    config = LlamaConfig.from_pretrained(base_model_path)
    config._attn_implementation = "eager"

    ar_model_local = LlamaForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()

    ar_tokenizer_local = AutoTokenizer.from_pretrained(base_model_path)

    # Wrap with Accelerator so it lives on GPU
    ar_model_prepared, ar_tokenizer_prepared = accelerator.prepare(
        ar_model_local, ar_tokenizer_local
    )

    ar_model = ar_model_prepared
    ar_tokenizer = ar_tokenizer_prepared

    return "✅ AR model loaded."

# -----------------------------------------------------------------------------
# 4) Load SPModel (delete AR model first)
# -----------------------------------------------------------------------------
def load_sp_model() -> str:
    """
    Delete any existing AR model (and free its GPU memory), then load SPModel onto GPU.
    """
    global ar_model, ar_tokenizer, sp_model, sp_tokenizer

    # If an AR model exists, move it to CPU, delete it, and free up CUDA
    if ar_model is not None:
        try:
            ar_model.cpu()
        except:
            pass
        try:
            del ar_model
            del ar_tokenizer
        except:
            pass

        torch.cuda.empty_cache()
        gc.collect()

        ar_model = None
        ar_tokenizer = None

    # Load a fresh SPModel
    base_sp = BASE_SP_MODEL
    draft_sp = DRAFT_SP_MODEL

    sp_model_local = SPModel.from_pretrained(
        base_model_path=base_sp,
        draft_model_path=draft_sp,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()

    sp_tokenizer_local = AutoTokenizer.from_pretrained(base_sp)

    sp_model_prepared, sp_tokenizer_prepared = accelerator.prepare(
        sp_model_local, sp_tokenizer_local
    )

    sp_model = sp_model_prepared
    sp_tokenizer = sp_tokenizer_prepared

    return "✅ SP model loaded."

# -----------------------------------------------------------------------------
# 5) SPModel GPU-warmup function (triggered by button)
# -----------------------------------------------------------------------------
def warm_up_sp_gpus(prompt: str) -> str:
    """
    Run 3 short SPModel generates on the current prompt, but do not stream.
    """
    global sp_model, sp_tokenizer
    if sp_model is None:
        return "⚠️ SPModel not loaded."

    # Tokenize & move to device
    encoded = sp_tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(accelerator.device)

    # Run 3 short generate passes
    for _ in range(3):
        gen = sp_model.spgenerate(
            input_ids,
            temperature=0.0,
            max_new_tokens=10,
            use_specextend=True,
            retrieval_chunk_size=32,
            retrieve_top_k=32,
            retrieve_every_n_steps=4
        )
        for _ in gen:
            pass

    return "✅ GPUs warmed up."

# -----------------------------------------------------------------------------
# 6) Streaming AR generation (all tokens black; do NOT re-display the prompt)
# -----------------------------------------------------------------------------
def stream_ar(user_message, history):
    """
    Streams naive autoregressive generation token-by-token.
    Yields [ new_history, tokens_per_sec, "" ] since AR has no avg_accept metric.
    """
    global ar_model, ar_tokenizer
    if ar_model is None:
        return

    # Immediately clear any auto-appended user message
    history = []

    # Initial placeholder (assistant only)
    yield [
        [ {"role": "assistant", "content": "🟡 Starting AR generation…"} ],
        "Tokens/sec: 0.0",
        ""
    ]

    # Tokenize & move to device
    encoded = ar_tokenizer(
        user_message,
        return_tensors="pt",
        add_special_tokens=True
    )
    input_ids = encoded["input_ids"].to(accelerator.device)

    past_key_values = None
    generated = input_ids.clone()
    t_start = time.time()
    total_tokens = 0
    eos_id = ar_tokenizer.eos_token_id

    output_buffer = ""

    with torch.no_grad():
        for step in range(256):
            if step == 0:
                out = ar_model(
                    input_ids=generated,
                    use_cache=True,
                    prefill=True
                )
            else:
                out = ar_model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            logits = out.logits
            past_key_values = out.past_key_values
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            total_tokens += 1
            elapsed = time.time() - t_start
            tps = total_tokens / (elapsed + 1e-8)

            token_str = ar_tokenizer.decode(
                next_token.squeeze(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Prepend a space if needed
            display_piece = token_str
            if token_str and not token_str.startswith(" "):
                display_piece = " " + token_str

            # All AR tokens in black
            output_buffer += f'<span style="color: black">{display_piece}</span>'

            # Emit the single assistant message as a list of one dict
            yield [
                [ {"role": "assistant", "content": output_buffer} ],
                f"Tokens/sec: {tps:.1f}",
                ""
            ]

            if next_token.item() == eos_id:
                break

# -----------------------------------------------------------------------------
# 7) Streaming SPModel (accepted→darker yellow "#D4AF37", resampled→black; do NOT re-display the prompt)
# -----------------------------------------------------------------------------
def internal_stream(user_message, history, use_specextend_flag):
    global sp_model, sp_tokenizer
    if sp_model is None:
        return

    # Immediately clear any auto-appended user message
    history = []

    # Initial placeholder (assistant only)
    yield [
        [ {"role": "assistant", "content": "🟡 Starting generation…"} ],
        "Tokens/sec: 0.0",
        "Avg Accept Len: 0.000"
    ]

    encoded = sp_tokenizer(
        user_message,
        return_tensors="pt",
        add_special_tokens=True
    )
    inputs = encoded["input_ids"].to(accelerator.device)

    t_start = time.time()
    total_tokens = 0

    stream_gen = sp_model.spgenerate(
        inputs,
        temperature=0.0,
        max_new_tokens=256,
        use_specextend=use_specextend_flag,
        retrieval_chunk_size=32,
        retrieve_top_k=32,
        retrieve_every_n_steps=4
    )

    output_buffer = ""
    for (token_str, token_type, avg_accept_length) in stream_gen:
        total_tokens += 1
        elapsed = time.time() - t_start
        tokens_per_sec = total_tokens / (elapsed + 1e-8)

        # accepted → darker yellow (#D4AF37); resampled → black
        color = "#D4AF37" if token_type == "accepted" else "black"
        prefix = "" if token_str.startswith(" ") else " "
        output_buffer += f'<span style="color: {color}">{prefix}{token_str}</span>'

        # Emit single assistant message
        yield [
            [ {"role": "assistant", "content": output_buffer} ],
            f"Tokens/sec: {tokens_per_sec:.1f}",
            f"Avg Accept Len: {avg_accept_length:.3f}"
        ]

def stream_no_specextend(user_message, history):
    for trip in internal_stream(user_message, history, use_specextend_flag=False):
        yield trip

def stream_with_specextend(user_message, history):
    for trip in internal_stream(user_message, history, use_specextend_flag=True):
        yield trip

# -----------------------------------------------------------------------------
# 8) Build the Gradio UI (use Default light theme)
# -----------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# SpecExtend Demo\n"
                "Settings: 1. Naive Autoregressive Generation  2. Standard Speculative Decoding (No SpecExtend)  3. Speculative Decoding with SpecExtend\n\n"
                "Base model: lmsys/vicuna-7b-v1.5-16k, Draft model: double7/vicuna-68m\n\n"
                "We use dynamic tree-based drafting for speculative decoding.\n\n"
                "### Steps:\n"
                "• Click **Load Sample** to load a prompt from GovReport or PG-19.\n\n"
                "• Click **Load AR Model** before generating autoregressively.\n\n"
                "• Click **Load SP Model** before generating with speculative decoding.\n\n"
                "• Press **Generate** under each chat to stream output tokens.\n\n")

    # Row: Load prompt
    with gr.Row():
        dataset_dropdown = gr.Dropdown(
            choices=[ "GovReport", "PG-19" ], 
            value="GovReport", 
            label="Dataset"
        )
        length_dropdown = gr.Dropdown(
            # “512” added before “1K”
            choices=["512", "1K", "2K", "4K", "8K", "16K"],
            value="2K",
            label="Max Length"
        )
        load_prompt_btn = gr.Button("Load Sample")
    prompt_box = gr.Textbox(placeholder="Loaded text appears here...", lines=4)

    load_prompt_btn.click(
        fn=load_and_warm,
        inputs=[dataset_dropdown, length_dropdown],
        outputs=[prompt_box]
    )

    # Row: Load models
    with gr.Row():
        load_ar_btn = gr.Button("Load AR Model")
        ar_status = gr.Markdown("AR not loaded.")
        load_sp_btn = gr.Button("Load SP Model")
        sp_status = gr.Markdown("SPModel not loaded.")

    load_ar_btn.click(fn=load_ar_model, inputs=None, outputs=[ar_status])
    load_sp_btn.click(fn=load_sp_model, inputs=None, outputs=[sp_status])

    # Row: Three chat columns
    with gr.Row():
        # Autoregressive column
        with gr.Column():
            gr.Markdown("### Autoregressive Generation")
            ar_tokens = gr.Markdown("Tokens/sec: 0.0")
            ar_accept = gr.Markdown("")  # no accept metric
            ar_chat = gr.Chatbot(type="messages", elem_id="ar_chat", height=400)
            ar_gen_btn = gr.Button("Generate (AR)")
            ar_gen_btn.click(
                fn=stream_ar,
                inputs=[prompt_box, ar_chat],
                outputs=[ar_chat, ar_tokens, ar_accept]
            )
            prompt_box.submit(
                fn=stream_ar,
                inputs=[prompt_box, ar_chat],
                outputs=[ar_chat, ar_tokens, ar_accept]
            )

        # SpecDec without SpecExtend column
        with gr.Column():
            gr.Markdown("### SpecDec (No SpecExtend)")
            no_tokens = gr.Markdown("Tokens/sec: 0.0")
            no_accept = gr.Markdown("Avg Accept Len: 0.000")
            no_chat = gr.Chatbot(type="messages", elem_id="no_chat", height=400)
            no_gen_btn = gr.Button("Generate (No SpecExtend)")
            no_gen_btn.click(
                fn=stream_no_specextend,
                inputs=[prompt_box, no_chat],
                outputs=[no_chat, no_tokens, no_accept]
            )
            prompt_box.submit(
                fn=stream_no_specextend,
                inputs=[prompt_box, no_chat],
                outputs=[no_chat, no_tokens, no_accept]
            )

        # SpecDec with SpecExtend column
        with gr.Column():
            gr.Markdown("### SpecDec (SpecExtend)")
            sp_tokens = gr.Markdown("Tokens/sec: 0.0")
            sp_accept = gr.Markdown("Avg Accept Len: 0.000")
            sp_chat = gr.Chatbot(type="messages", elem_id="sp_chat", height=400)

            # New "Warm up GPUs" button and status
            sp_warm_btn = gr.Button("Warm up GPUs")
            sp_warm_status = gr.Markdown("GPUs not warmed.")

            sp_warm_btn.click(
                fn=warm_up_sp_gpus,
                inputs=[prompt_box],
                outputs=[sp_warm_status]
            )

            sp_gen_btn = gr.Button("Generate (SpecExtend)")
            sp_gen_btn.click(
                fn=stream_with_specextend,
                inputs=[prompt_box, sp_chat],
                outputs=[sp_chat, sp_tokens, sp_accept]
            )
            prompt_box.submit(
                fn=stream_with_specextend,
                inputs=[prompt_box, sp_chat],
                outputs=[sp_chat, sp_tokens, sp_accept]
            )

    # demo.launch()
    demo.launch(share=True)