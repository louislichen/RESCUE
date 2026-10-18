import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .config import SystemConfig
from .engine import RescueEngine
from .loading import (DEFAULT_COST, DEFAULT_MAPPER, DEFAULT_RISK, DEFAULT_STUDENT,
                      DEFAULT_TEACHER, load_models, load_modules)


def main():
    parser = argparse.ArgumentParser(description='Attention-guided exact greedy speculative decoding')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--prompt-file', type=Path)
    parser.add_argument('--teacher', default=DEFAULT_TEACHER)
    parser.add_argument('--student', default=DEFAULT_STUDENT)
    parser.add_argument('--mapper', type=Path, default=DEFAULT_MAPPER)
    parser.add_argument('--risk', type=Path, default=DEFAULT_RISK)
    parser.add_argument('--cost', type=Path, default=DEFAULT_COST)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--selector', choices=['coverage', 'joint', 'teacher', 'read_joint', 'full'], default='coverage')
    parser.add_argument('--schedule', choices=['auto', 'serial', 'async'], default='auto')
    parser.add_argument('--attention-backend', choices=['eager', 'sdpa'], default='sdpa')
    parser.add_argument('--refresh-every', type=int, default=4)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--recent', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--prefill-chunk-size', type=int, default=0)
    parser.add_argument('--length-endpoint', help='HOST:PORT for a separate CPU length service')
    parser.add_argument('--no-injection', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Also generate an independent Target reference')
    args = parser.parse_args()
    endpoint = None
    if args.length_endpoint:
        host, port = args.length_endpoint.rsplit(':', 1)
        endpoint = (host, int(port))
    config = SystemConfig(selector=args.selector, schedule=args.schedule, attention_backend=args.attention_backend,
                          refresh_every=args.refresh_every, budget=args.budget, recent=args.recent,
                          chunk_size=args.chunk_size, max_new_tokens=args.max_new_tokens,
                          prefill_chunk_size=args.prefill_chunk_size,
                          length_endpoint=endpoint, inject_tokens=0 if args.no_injection else 128)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher, student, tokenizer = load_models(args.teacher, args.student, args.device)
    mapper, risk, cost = load_modules(args.device, args.mapper, args.risk, args.cost)
    text = args.prompt if args.prompt is not None else args.prompt_file.read_text()
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    with RescueEngine(teacher, student, config, mapper, risk, cost, vocab_size=len(tokenizer)) as engine:
        reference = engine.reference(ids, tokenizer.eos_token_id) if args.verify else None
        result = engine.generate(ids, tokenizer.eos_token_id)
    result['text'] = tokenizer.decode(result['generated'], skip_special_tokens=True)
    result['config'] = asdict(config)
    if reference is not None:
        result['output_agrees'] = result['generated'] == reference['generated']
        if not result['output_agrees']:
            raise AssertionError('Output differs from Target reference')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
