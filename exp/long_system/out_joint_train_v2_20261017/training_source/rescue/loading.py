"""Explicit local checkpoints; base weights are never downloaded implicitly."""
from pathlib import Path
import hashlib
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp.selective_kv.bridge import ValueMapper
from .policy import RiskModel, load_cost

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = '/mnt/8T/cl/models/Qwen2-7B-Instruct'
DEFAULT_STUDENT = '/mnt/8T/cl/models/Qwen2-1.5B-Instruct'
DEFAULT_MAPPER = ROOT/'exp/selective_kv/out_pilot/performance_gated.pt'
DEFAULT_RISK = ROOT/'exp/attention_horizon/out/risk_attention_read_42.pt'
DEFAULT_COST = ROOT/'exp/attention_horizon/out/cost_model.json'


def load_models(teacher=DEFAULT_TEACHER, student=DEFAULT_STUDENT, device='cuda:0',
                target_dtype=torch.float32, efficient=False):
    tokenizers, models = [], []
    for path, dtype in ((teacher, target_dtype), (student, torch.bfloat16)):
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, local_files_only=True,
                                                   attn_implementation='eager').to(device).eval()
        tokenizers.append(tokenizer)
        models.append(model)
    if efficient:
        from .efficient_attention import install_efficient_attention
        for model in models:
            install_efficient_attention(model)
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
        raise ValueError('Target and Draft must share a token vocabulary')
    return (*models, tokenizers[0])


def calibrated_risk(risk, checkpoint, calibration=None):
    if calibration is None:
        return risk
    payload = json.loads(Path(calibration).read_text())
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != payload['base_checkpoint_sha256']:
        raise ValueError('Hazard calibration is bound to another risk checkpoint')
    return risk.calibrated(payload['scale'], payload['accept_bias'])


def load_modules(device, mapper=DEFAULT_MAPPER, risk=DEFAULT_RISK, cost=DEFAULT_COST, calibration=None):
    payload = torch.load(mapper, map_location='cpu', weights_only=False)
    network = ValueMapper(**payload['config']).to(device)
    network.load_state_dict(payload['state_dict'])
    policy = calibrated_risk(RiskModel.load(risk, device), risk, calibration)
    return network.eval().requires_grad_(False), policy, load_cost(cost)
