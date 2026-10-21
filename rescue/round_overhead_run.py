"""Frozen, paired execution-only comparison; no training or paused-test resume.

The baseline package is imported under a separate name from an explicit root.
Canonical outputs and instrumented equality checks precede observer-free timing.
"""
import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict

import torch


BASELINE_COMMIT = '24f63f987f36fe167620e2a418d57574860cf7df'
MAPPER_SHA256 = 'd08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f'
POLICY_SHA256 = '3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5'
SOURCE_INDICES = (303, 306, 327, 604, 613, 614)
LENGTHS = (4096, 8192)
REPEATS = 3
MAX_NEW_TOKENS = 128
BASELINE_PINS = {
    'engine.py': 'b8c5751f224e780d02194fb022676131e20a50b37902dcf6020b37b4dc87c86c',
    'features.py': '670b2fcd45d20d9e117ca624a5b78c42e351fcf18d7ff7d54794f7f69ea77fbf',
    'selection.py': '9141d54a76aa6998c1760349b8d11818558656ff05cfdc18f30a63e613237c49',
    'step_budget.py': '8dbb844c7a8915f08e5c4b96af854517e3f804a4cf4e675dd26906ec8c5a0f54',
}
SHARED_EXP_FILES = ('exp/attention_horizon/selection.py', 'exp/selective_kv/bridge.py')
TRACE_KEYS = ('round_index', 'prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, payload):
    with Path(path).open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def append(path, payload):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(payload, allow_nan=False)+'\n')
        handle.flush(); os.fsync(handle.fileno())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def context(item, length):
    prefix, suffix = item['prefix_ids'], item['suffix_ids']
    count = length-len(prefix)-len(suffix)
    require(0 < count <= len(item['document_ids']), 'Natural document is too short')
    ids = prefix+item['document_ids'][:count]+suffix
    require(len(ids) == length and all(type(v) is int and v >= 0 for v in ids), 'Invalid exact token input')
    return ids


def read_documents(paths):
    documents = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            index = item['source_index']
            require(index in SOURCE_INDICES and index not in documents, 'Wrong/duplicate registered document')
            require(item.get('split') != 'test', 'This runner must not open a test split')
            digest = item.get('document_sha256', '')
            require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Missing document SHA256')
            for length in LENGTHS:
                context(item, length)
            documents[index] = item
    require(set(documents) == set(SOURCE_INDICES), 'Exactly the six registered documents are required')
    require(len({v['document_sha256'] for v in documents.values()}) == len(SOURCE_INDICES), 'Duplicate document body')
    return documents


def timing_schedule(seed):
    cells = [(i, n) for i in SOURCE_INDICES for n in LENGTHS]
    rng = random.Random(seed)
    starts = [True]*6+[False]*6; rng.shuffle(starts)
    initial = dict(zip(cells, starts))
    plan, pair = [], 0
    for repeat in range(REPEATS):
        order = list(cells); rng.shuffle(order)
        for index, length in order:
            first_before = initial[index, length] != bool(repeat % 2)
            arms = ('before', 'after') if first_before else ('after', 'before')
            for mode in arms:
                plan.append(dict(pair_index=pair, source_index=index, length=length,
                                 repeat=repeat, mode=mode))
            pair += 1
    return plan


def trace_signature(row):
    return [{key: trace[key] for key in TRACE_KEYS} for trace in row['trace']]


def check_reference(row, reference):
    require(row['generated'] == reference, 'Generated tokens differ from independent Target reference')
    require(row['tokens'] == len(reference) and 0 < row['tokens'] <= MAX_NEW_TOKENS, 'Invalid output accounting')
    require(row['accepted'] == sum(t['accepted'] for t in row['trace']), 'Accepted-token accounting drift')
    require(row['proposed'] == sum(t['proposed'] for t in row['trace']), 'Proposed-token accounting drift')
    require(row['blocks'] == len(row['trace']), 'Round accounting drift')
    row['output_agrees'] = True


def check_pair(before, after):
    require(before['generated'] == after['generated'], 'Paired generated tokens differ')
    require(trace_signature(before) == trace_signature(after), 'Paired depth/acceptance trajectory differs')
    for key in ('tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes'):
        require(before[key] == after[key], f'Paired {key} differs')


class CorrectnessObserver:
    step_budget = True

    def __init__(self):
        self.rounds, self.nonzero_injection = [], False

    def __call__(self, packet):
        vector = packet['budget_features']
        values = None if vector is None else vector.detach().float().cpu().tolist()
        require(values is None or (len(values) == 12 and all(math.isfinite(v) for v in values)), 'Invalid causal features')
        if values is not None and values[-1] > 0:
            self.nonzero_injection = True
        self.rounds.append(dict(round_index=packet['round_index'], prefix=packet['prefix'],
            progress=packet['progress'], remaining=packet['remaining'],
            selected=packet['selected'].detach().cpu().tolist(), budget_features=values,
            draft=list(packet['draft']), checks=list(packet['checks']), accepted=packet['accepted']))


def alias_package(root, name='_rescue_round_overhead_before'):
    path = Path(root).resolve()/'rescue'/'__init__.py'
    require(path.is_file(), 'Baseline rescue package is missing')
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve() == path, 'Baseline alias is already bound elsewhere')
    else:
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return {part: importlib.import_module(name+'.'+part) for part in
            ('engine', 'config', 'query_mapper', 'step_budget', 'features', 'selection')}


def source_inventory(root):
    paths = sorted((root/'rescue').glob('*.py'))+[root/name for name in SHARED_EXP_FILES]
    require(all(p.is_file() for p in paths), 'Incomplete source tree')
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def validate_sources(before_root, after_root):
    before, after = source_inventory(before_root), source_inventory(after_root)
    for name, digest in BASELINE_PINS.items():
        require(before.get('rescue/'+name) == digest, f'Baseline is not the pinned commit: {name}')
    for name, digest in before.items():
        if name not in {'rescue/'+n for n in BASELINE_PINS}:
            require(after.get(name) == digest, f'Unexpected non-treatment runtime source change: {name}')
    return dict(before=before, after=after)


def check_module_paths(old, before_root, after_root):
    for name in BASELINE_PINS:
        part = name[:-3]
        old_module = old[part]
        require(Path(old_module.__file__).resolve() == before_root/'rescue'/name, 'Baseline alias escaped its source root')
        require(sha256(old_module.__file__) == BASELINE_PINS[name], 'Baseline module source binding changed')
        new = importlib.import_module('rescue.'+part)
        require(Path(new.__file__).resolve() == after_root/'rescue'/name, 'Optimized module escaped its source root')
    for relative in SHARED_EXP_FILES:
        module = importlib.import_module(relative[:-3].replace('/', '.'))
        require(Path(module.__file__).resolve() == after_root/relative, 'Shared exp dependency escaped optimized root')
        require(sha256(module.__file__) == sha256(before_root/relative), 'Shared exp dependency differs across arms')
    alias = old['engine'].__package__
    for name, module in list(sys.modules.items()):
        if name == alias or name.startswith(alias+'.'):
            path = getattr(module, '__file__', None)
            require(path is not None, 'Baseline module has no auditable source path')
            try:
                Path(path).resolve().relative_to(before_root/'rescue')
            except ValueError:
                raise RuntimeError('Baseline submodule escaped isolated source root: '+name)
    require(not hasattr(old['step_budget'].StepBudget, 'decide_cap'), 'Baseline unexpectedly contains the fast API')


def model_files(root):
    suffixes = {'.json', '.model', '.safetensors', '.bin', '.txt'}
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in suffixes
                   and not any(part.startswith('.') for part in p.relative_to(root).parts))
    require(any(p.suffix in {'.safetensors', '.bin'} for p in paths), 'Local model weights are missing')
    require((root/'config.json').is_file(), 'Model configuration is missing')
    return paths


def gpu_snapshot():
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Require one visible CUDA GPU')
    props = torch.cuda.get_device_properties(0)
    require('H100' in props.name, 'This registered execution experiment requires H100')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                   '--format=csv,noheader,nounits'], text=True)
    process_raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                           '--format=csv,noheader,nounits'], text=True)
    other = []
    for line in process_raw.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if len(fields) >= 2 and fields[1].isdigit() and int(fields[1]) != os.getpid():
            other.append(line)
    require(not other, 'GPU conflict: other compute processes are active; no job was interrupted: '+repr(other))
    return dict(gpu_name=props.name, total_memory=props.total_memory,
                gpu_uuid=str(getattr(props, 'uuid', 'unavailable')), nvidia_smi_gpu=raw.strip(),
                nvidia_smi_compute=process_raw.strip(), pid=os.getpid(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                torch=str(torch.__version__), cuda=torch.version.cuda, target_dtype='torch.float32',
                draft_dtype='torch.bfloat16', prefix_dtype='torch.bfloat16', tf32=False)


def run_experiment(args):
    from .config import SystemConfig
    from .engine import RescueEngine
    from .baselines import BaselineEngine
    from .loading import load_models
    from .packing import pack_projections
    from .query_mapper import mapper_from_payload
    from .step_budget import StepBudget

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    require(not output.exists(), 'Fresh output required; partial runs must not be reused')
    for name in ('baseline_root', 'teacher', 'student'):
        protected = getattr(args, name, None)
        if protected is not None:
            require(not output.is_relative_to(protected.resolve()),
                    'Output must not be placed inside baseline or base-model assets')
    output.mkdir(parents=True)
    counts = dict(reference_requests=0, correctness_pairs=0, warmup_requests=0, timing_requests=0)
    try:
        for path in (args.baseline_root, args.teacher, args.student, args.mapper, args.policy, args.cost,
                     args.warmup_file, *args.data_files):
            require(path.is_absolute() and path.exists(), 'All asset/source/data paths must be explicit existing absolute paths')
        before_root = args.baseline_root.resolve()
        require(before_root != root, 'Baseline and optimized source roots must be isolated')
        sources = validate_sources(before_root, root)
        old = alias_package(before_root)
        check_module_paths(old, before_root, root)
        require(sha256(args.mapper) == MAPPER_SHA256, 'Mapper differs from the fixed nonzero checkpoint')
        require(sha256(args.policy) == POLICY_SHA256, 'Length policy differs from the fixed StepBudget checkpoint')
        documents = read_documents(args.data_files)
        warm_rows = [json.loads(l) for l in args.warmup_file.read_text().splitlines() if l.strip()]
        require(len(warm_rows) == 1, 'Exactly one explicit training warmup document is required')
        warmup = warm_rows[0]
        require(warmup['source_index'] not in SOURCE_INDICES and warmup.get('split') != 'test', 'Warmup must not reuse confirmation/test')
        for n in LENGTHS:
            context(warmup, n)
        cfg = SystemConfig(selector='hierarchical', coverage_rows=64, length_policy='risk',
            fixed_k=4, injection_layers=(27,), inject_tokens=64, prefill_precision='bfloat16',
            max_new_tokens=MAX_NEW_TOKENS, optimized_execution=True, refresh_every=4,
            refresh_policy='periodic', schedule='serial', max_draft=12, profile=False)
        before_cfg = old['config'].SystemConfig(**asdict(cfg))
        cost = json.loads(args.cost.read_text())
        mapper_payload = torch.load(args.mapper, map_location='cpu', weights_only=True)
        policy_payload = torch.load(args.policy, map_location='cpu', weights_only=True)
        require(mapper_payload['config'].get('kind') in ('query_residual', 'query_attention_residual', 'native_residual_query'), 'Expected V-only residual mapper')
        require(policy_payload.get('kind') == 'step_budget', 'Expected the fixed StepBudget policy')
        paths = sorted(set(model_files(args.teacher)+model_files(args.student)+[
            args.mapper, args.policy, args.cost, args.warmup_file, *args.data_files]))
        assets = {str(p): sha256(p) for p in paths}
        for mode, inventory in sources.items():
            source_root = before_root if mode == 'before' else root
            for relative in inventory:
                destination = output/'source'/mode/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root/relative, destination)
        torch.set_num_threads(4); torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        environment = gpu_snapshot()
        teacher, student, tokenizer = load_models(str(args.teacher), str(args.student), device='cuda:0',
                                                   target_dtype=torch.float32, efficient=True)
        pack_projections(teacher); pack_projections(student)
        mapper_before = old['query_mapper'].mapper_from_payload(mapper_payload, 'cuda:0').eval().requires_grad_(False)
        mapper_after = mapper_from_payload(mapper_payload, 'cuda:0').eval().requires_grad_(False)
        policies = dict(before=old['step_budget'].StepBudget.from_payload(policy_payload, 'cuda:0'),
                        after=StepBudget.from_payload(policy_payload, 'cuda:0'))
        mappers = dict(before=mapper_before, after=mapper_after)
        for model in (teacher, student, *mappers.values(), *policies.values()):
            model.eval().requires_grad_(False)
            require(all(p.device == torch.device('cuda:0') and not p.requires_grad for p in model.parameters()), 'All frozen components must reside on cuda:0')
        require(teacher.dtype == torch.float32 and student.dtype == torch.bfloat16, 'Unexpected base precision')
        require(27 in mapper_after.layers, 'Fixed last-layer injection route is absent')
        relative = mapper_after.layers.index(27)
        nonzero_head = any(torch.count_nonzero(p).item() for block in
                          (mapper_after.decoders[relative], mapper_after.residuals[relative][-1]) for p in block.parameters())
        require(nonzero_head, 'Active mapper correction heads are all zero')
        plan = timing_schedule(args.seed)
        protocol = dict(schema=1, baseline_commit=BASELINE_COMMIT,
            scope='Execution-only paired validation of three overhead changes; no training, new algorithm, or SpecExtend superiority claim',
            source_indices=list(SOURCE_INDICES), lengths=list(LENGTHS), repeats=REPEATS,
            max_new_tokens=MAX_NEW_TOKENS, max_draft=12, seed=args.seed, config=asdict(cfg),
            environment=environment, mapper_nonzero=True, mapper_frozen=True,
            request_buffer_policy='cold_request', source_sha256=sources, assets_sha256=assets,
            expected_mapper_sha256=MAPPER_SHA256, expected_policy_sha256=POLICY_SHA256,
            args={k: [str(p) for p in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            timing_order=plan, expected_timing_requests=72, expected_correctness_pairs=12,
            expected_reference_requests=12, expected_warmup_requests=4,
            timing='engine.generate whole request includes buffer setup, prefill, decode and cleanup; constructors/model load/audits/observers excluded',
            cohort='Six reused development-confirmation documents, not a fresh generalization test; 4K/8K nested prefixes',
            observation='Instrumented correctness runs excluded from timing; timing has no observers and profile=False')
        write(output/'protocol.json', protocol)
        configs = dict(before=before_cfg, after=cfg)
        engines = dict(before=old['engine'].RescueEngine, after=RescueEngine)

        def run(item, length, mode, observer=None):
            gc.collect()
            with engines[mode](teacher, student, configs[mode], mappers[mode], policies[mode], cost,
                               vocab_size=len(tokenizer), round_observer=observer) as engine:
                row = engine.generate(context(item, length), tokenizer.eos_token_id)
            row.update(mode=mode, source_index=item['source_index'], length=length)
            return row

        for length in LENGTHS:
            warm = {}
            for mode in ('before', 'after'):
                row = run(warmup, length, mode); warm[mode] = row
                append(output/'warmup.jsonl', row); counts['warmup_requests'] += 1
            check_pair(warm['before'], warm['after'])
        references = {}
        reference_cfg = SystemConfig(**{**asdict(cfg), 'length_policy': 'fixed'})
        for index in SOURCE_INDICES:
            for length in LENGTHS:
                with BaselineEngine(teacher, student, reference_cfg, vocab_size=len(tokenizer)) as engine:
                    row = engine.reference(context(documents[index], length), tokenizer.eos_token_id)
                require(0 < len(row['generated']) <= MAX_NEW_TOKENS, 'Invalid canonical output length')
                key = f'{index}:{length}'; references[key] = row['generated']
                row.update(source_index=index, length=length, mode='target_ar')
                append(output/'reference_requests.jsonl', row); counts['reference_requests'] += 1
        write(output/'references.json', references)
        for index in SOURCE_INDICES:
            for length in LENGTHS:
                pair, observed = {}, {}
                for mode in ('before', 'after'):
                    observer = CorrectnessObserver()
                    row = run(documents[index], length, mode, observer)
                    row['output_agrees'] = row['generated'] == references[f'{index}:{length}']
                    append(output/'correctness_requests.jsonl', row)
                    check_reference(row, references[f'{index}:{length}'])
                    require(observer.nonzero_injection, 'No actual nonzero injection observed in correctness request')
                    pair[mode], observed[mode] = row, observer
                check_pair(pair['before'], pair['after'])
                require(observed['before'].rounds == observed['after'].rounds, 'Selected positions/causal features/draft/checks differ')
                append(output/'correctness.jsonl', dict(source_index=index, length=length, passed=True,
                    output_agrees=True, trajectory_equal=True, rounds_equal=True,
                    nonzero_injection={mode: observed[mode].nonzero_injection for mode in observed},
                    **{mode: dict(generated=pair[mode]['generated'], trace_signature=trace_signature(pair[mode]),
                                  rounds=observed[mode].rounds) for mode in pair}))
                counts['correctness_pairs'] += 1
                print('correctness passed', index, length, flush=True)
        # All source and immutable assets are rechecked immediately before the
        # first timing request, outside either arm's clock.
        require(validate_sources(before_root, root) == sources, 'Runtime sources changed before timing')
        require({str(p): sha256(p) for p in paths} == assets, 'Assets changed before timing')
        write(output/'PRE_TIMING.json', dict(passed=True, counts=counts.copy(), environment=gpu_snapshot()))
        pending = {}
        for entry in plan:
            # Do not compete with another job that appeared after preflight.
            gpu_snapshot()
            row = run(documents[entry['source_index']], entry['length'], entry['mode'])
            row.update(entry)
            row['output_agrees'] = row['generated'] == references[f"{entry['source_index']}:{entry['length']}"]
            append(output/'runtime.jsonl', row); counts['timing_requests'] += 1
            check_reference(row, references[f"{entry['source_index']}:{entry['length']}"])
            pending[entry['mode']] = row
            if len(pending) == 2:
                check_pair(pending['before'], pending['after']); pending = {}
                print('timing pair passed', entry['pair_index'], counts['timing_requests'], flush=True)
        require(not pending and counts == dict(reference_requests=12, correctness_pairs=12,
            warmup_requests=4, timing_requests=72), 'Registered matrix incomplete')
        require(validate_sources(before_root, root) == sources, 'Sources changed during timing')
        require({str(p): sha256(p) for p in paths} == assets, 'Immutable assets changed during timing')
        write(output/'COMPLETE.json', dict(passed=True, **counts, assets_unchanged=True,
            expected_timing_requests=72, source_unchanged=True, mapper_nonzero=True,
            analysis_required=True, scope='Execution matrix complete; figures/report required before experiment handoff'))
    except BaseException as error:
        write(output/'FAILURE.json', dict(passed=False, error_type=type(error).__name__,
            error=str(error), counts=counts, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            policy='Stopped without restart, weight/data changes, cleanup or other-job interruption'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-root', 'teacher', 'student', 'mapper', 'policy', 'cost', 'warmup-file', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--data-files', type=Path, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=619)
    run_experiment(parser.parse_args())


if __name__ == '__main__':
    main()
