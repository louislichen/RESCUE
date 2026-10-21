"""V3 candidate-analytic smoke contract correction, not reference equivalence.

All112 fixtures and all arithmetic remain exactly v2. Candidate known values
still require bitwise equality; reference known values are diagnostic only.
V1 and v2 both terminalFAIL records must remain preserved and cannot be resumed
or relabelled. V2 zero-Q/K did not change the observed one-key reference error:
that proposed explanation is falsified, not a reason for more fixture changes.
Only the mandatory reference-analytic requirement and explicitly scoped labels
change; candidate algorithm, random tolerances and final Token gate do not.
This standalone smoke cannot establish numerical/reference equivalence, Target
Token equality, model quality or performance. Frozen sources are read-only.

Default PLAN_ONLY; a future explicit standalone CUDA operator smoke gate.

NO engine/model/corpus/TPS/Token gate. No random allclose tolerance is defined.
Only --execute-cuda may import Torch/Triton or create output. This preparation
does not authorize running it; use a fresh owned subprocess on an externally
verified idle GPU. No remote launch, retries, fallback or cleanup is provided.

Reference: pinned Torch2.7.1 FP32 folded F.scaled_dot_product_attention under
sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]), never our own math reference.
https://docs.pytorch.org/docs/2.7/generated/torch.nn.attention.sdpa_kernel.html
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT/'exp/round_overhead/fp32_splitkv_candidate_v1.py'
CANDIDATE_TEST = ROOT/'tests/test_fp32_splitkv_candidate_v1.py'
CANDIDATE_SHA = '9f133cc824915afd09d9ad25a1b16e6cd721522e1f583cabd5e102235358cfd2'
CANDIDATE_TEST_SHA = 'd05aab7170f75451f1af4e93aac9a895c25e9b3f15ae9c59d11e2b6117bed8c0'
GPU_UUID = 'GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9'
TORCH_VERSION, TORCH_CUDA_VERSION, TRITON_VERSION = '2.7.1+cu128', '12.8', '3.3.1'
V1_GATE = ROOT/'exp/round_overhead/fp32_splitkv_cuda_gate_v1.py'
V1_GATE_TEST = ROOT/'tests/test_fp32_splitkv_cuda_gate_v1.py'
V1_GATE_SHA = '62957e77c8d26b61dd89fc7692189ff0f46d61f3d77b9f6131b996eae26de98d'
V1_GATE_TEST_SHA = 'ae1f6665852f99f947c95ade0c7f766186fb80615b1260911a5f8b27aada78ef'
V2_GATE = ROOT/'exp/round_overhead/fp32_splitkv_cuda_gate_v2.py'
V2_GATE_TEST = ROOT/'tests/test_fp32_splitkv_cuda_gate_v2.py'
V2_GATE_SHA = '1d4b340786851eb41c80489fb6c3b775101ba8125e97d50e81903d4536a52549'
V2_GATE_TEST_SHA = '525ad7ade08a3fff6c8b593c88d8d32bc4d7f3399de35025f48146b4f1ff02b5'
FIXTURE_VERSION = 2
CONTRACT_VERSION = 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def source_pins():
    result = dict(candidate_sha256=sha(CANDIDATE), candidate_test_sha256=sha(CANDIDATE_TEST),
                  gate_sha256=sha(Path(__file__)), fixture_version=FIXTURE_VERSION, contract_version=CONTRACT_VERSION,
                  v1_gate_sha256=sha(V1_GATE), v1_gate_test_sha256=sha(V1_GATE_TEST),
                  v2_gate_sha256=sha(V2_GATE), v2_gate_test_sha256=sha(V2_GATE_TEST))
    require(result['v1_gate_sha256'] == V1_GATE_SHA and result['v1_gate_test_sha256'] == V1_GATE_TEST_SHA,
            'Frozen v1 gate/proof source changed')
    require(result['v2_gate_sha256'] == V2_GATE_SHA and result['v2_gate_test_sha256'] == V2_GATE_TEST_SHA,
            'Frozen v2 gate/proof source changed')
    require(result['candidate_sha256'] == CANDIDATE_SHA, 'Frozen candidate source changed')
    require(result['candidate_test_sha256'] == CANDIDATE_TEST_SHA, 'Frozen candidate CPU proof changed')
    return result


def cases():
    result = []
    for q in range(2, 6):
        for n in (2048, 4096, 6144, 8192, 12288, 16384):
            for mask in ('bool_broadcast', 'additive_broadcast', 'bool_full', 'additive_full'):
                result.append(dict(kind='random', q=q, n=n, mask=mask, exact=False))
    for q in range(2, 6):
        result.extend((dict(kind='zero_scores', q=q, n=16512 if q == 2 else 2048, mask='none', exact=True),
                       dict(kind='constant_v', q=q, n=2048, mask='none', exact=True),
                       dict(kind='all_masked', q=q, n=q, mask='bool_broadcast' if q % 2 == 0 else 'additive_broadcast', exact=True),
                       dict(kind='one_key', q=q, n=129 if q < 5 else 1025,
                            allowed_key={2: 0, 3: 127, 4: 128, 5: 1024}[q],
                            mask='bool_broadcast' if q % 2 == 0 else 'additive_broadcast', exact=True)))
    for index, case in enumerate(result):
        case.update(case_id=f'operator_{index:03d}', seed=20261021+index,
                    query_layout='noncontiguous_folded', kv_head_stride=0)
    require(len(result) == 112, 'Fixed112 operator cases changed')
    return result


def plan():
    return dict(status='PLAN_ONLY', passed=False, actual_cuda=False, new_cuda_calls=0,
        fixture_version=FIXTURE_VERSION, contract_version=CONTRACT_VERSION,
        v1_gate_sha256=V1_GATE_SHA, v1_gate_test_sha256=V1_GATE_TEST_SHA,
        v2_gate_sha256=V2_GATE_SHA, v2_gate_test_sha256=V2_GATE_TEST_SHA,
        fixtures='All112 exactly v2; no new case selection, arithmetic, V, mask or RNG changes',
        contract_only_change='Candidate analytic bitwise mandatory; reference analytic bitwise diagnostic only',
        prior_failures='V1 and v2 terminalFAIL preserved separately; never resume, overwrite or relabel',
        zero_qk_hypothesis='V2 zero Q/K left one-key reference error unchanged; hypothesis falsified, no fixture roulette',
        candidate_analytic_smoke_passed=False, passed_scope='CANDIDATE_ANALYTIC_ONLY',
        reference_equivalence_gate='NOT_ESTABLISHED_DIAGNOSTIC_ONLY',
        reference_analytic_gate='DIAGNOSTIC_ONLY',
        actual_cuda_definition='At least one reference+candidate pair returned and explicit CUDA synchronization succeeded; availability alone is not execution',
        numerical_pass=False, quality_gate='NOT_RUN', performance_gate='NOT_RUN', model_gate='NOT_RUN', token_gate='NOT_RUN',
        candidate_source_sha256=CANDIDATE_SHA, candidate_test_source_sha256=CANDIDATE_TEST_SHA,
        gpu_uuid=GPU_UUID, torch_version=TORCH_VERSION, torch_cuda_version=TORCH_CUDA_VERSION, triton_version=TRITON_VERSION,
        reference_backend='Torch2.7.1 EFFICIENT_ATTENTION only', dtype='float32', tf32=False,
        splits=8, block_keys=128, cases=cases(), exact_cases=16, random_diagnostic_cases=96,
        random_tolerance=None, random_status='DIAGNOSTIC_COMPLETE only; no numerical-equivalence pass',
        future_requirements='Human authorization plus explicit --execute-cuda, fresh owned subprocess/output/cache, externally idle fixed GPU')


def _owned_directory(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, 'Existing canonical absolute directory required')
    info = path.stat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o022,
            'Directory must be user-owned without group/world write access')
    require(path not in (Path('/'), Path.home().resolve()), 'Broad directory forbidden')
    return path


def preflight_paths(output, cache):
    require(output is not None and cache is not None, '--output and --cache-dir are mandatory for execution')
    output = Path(output)
    require(output.is_absolute() and _owned_directory(output.parent)/output.name == output,
            'Output must have an existing owned canonical parent')
    require(not os.path.lexists(output), 'Fresh nonexistent output required; never resume or overwrite')
    cache = _owned_directory(cache)
    require(not any(cache.iterdir()), 'Fresh empty owned cache required')
    require(not output.is_relative_to(cache) and not cache.is_relative_to(output), 'Output/cache must be disjoint')
    require(not output.is_relative_to(ROOT) and not cache.is_relative_to(ROOT), 'No output/cache writes inside the source tree')
    require(os.environ.get('TRITON_CACHE_DIR') == str(cache), 'TRITON_CACHE_DIR must already name owned cache')
    require(os.environ.get('CUDA_CACHE_DISABLE') == '1', 'CUDA_CACHE_DISABLE=1 required')
    require('TRITON_CACHE_MANAGER' not in os.environ, 'Custom/empty cache manager override forbidden')
    for name in ('TRITON_REMOTE_CACHE_BACKEND', 'TRITON_DUMP_DIR', 'TRITON_OVERRIDE_DIR'):
        require(not os.environ.get(name), f'{name} override forbidden')
    for name in ('TRITON_INTERPRET', 'TRITON_KERNEL_DUMP', 'TRITON_KERNEL_OVERRIDE'):
        require(os.environ.get(name, '0') in ('', '0'), f'{name} must be disabled')
    return output, cache


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _canonical_uuid(value):
    value = str(value)
    bare = value[4:] if value.startswith('GPU-') else value
    normalized = str(uuid.UUID(bare))
    require(bare.lower() == normalized, 'Noncanonical GPU UUID')
    return 'GPU-'+normalized


def _inputs(torch, case):
    q, n = case['q'], case['n']
    generator = torch.Generator(device='cpu').manual_seed(case['seed'])
    rand = lambda shape: torch.randn(shape, generator=generator, dtype=torch.float32)
    query_base = rand((1, q, 28, 128))
    key_base, value_base = rand((4, 1, n, 128)), rand((4, 1, n, 128))
    if case['kind'] == 'one_key':
        query_base.zero_()
        key_base.zero_()
    expected = None
    if case['kind'] in ('zero_scores', 'constant_v'):
        query_base.zero_(); key_base.zero_()
        if case['kind'] == 'zero_scores':
            value_base[:] = (1-2*(torch.arange(n) % 2)).float()[None, None, :, None]
            expected = torch.zeros((4, 7, q, 128), dtype=torch.float32)
        else:
            constants = torch.pow(torch.tensor(2., dtype=torch.float32), torch.arange(128) % 7-3)
            value_base[:] = constants
            expected = constants[None, None, None].expand(4, 7, q, 128).contiguous()
    query = query_base.to('cuda:0').transpose(1, 2).reshape(4, 7, q, 128)
    key = key_base.to('cuda:0').expand(4, 7, n, 128)
    value = value_base.to('cuda:0').expand(4, 7, n, 128)
    mask = None
    if case['mask'] != 'none':
        shape = (4, 7, q, n) if case['mask'].endswith('full') else (1, 1, q, n)
        allowed = torch.rand(shape, generator=generator) > .15
        allowed[..., 0] = True
        if case['kind'] == 'all_masked':
            allowed.zero_()
            expected = torch.zeros((4, 7, q, 128), dtype=torch.float32)
        elif case['kind'] == 'one_key':
            allowed.zero_(); allowed[..., case['allowed_key']] = True
            expected = value_base[:, :, case['allowed_key']:case['allowed_key']+1].expand(4, 7, q, 128).contiguous()
        mask = (allowed if case['mask'].startswith('bool') else
                torch.zeros(shape, dtype=torch.float32).masked_fill_(~allowed, -float('inf'))).to('cuda:0')
    require(not query.is_contiguous() and key.stride(1) == value.stride(1) == 0, 'Prescribed view layout changed')
    return query, key, value, mask, None if expected is None else expected.to('cuda:0')


def _snapshot(torch, inputs):
    result = []
    for tensor in inputs:
        if tensor is None:
            result.append(None); continue
        # The whole actual backing allocation is valid for these gate-owned
        # tensors. Hash it once, not the sevenfold expanded logical KV view.
        nbytes = tensor.untyped_storage().nbytes()
        view = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(tensor.untyped_storage(), 0, (nbytes,), (1,))
        value = hashlib.sha256(view.cpu().numpy().tobytes()).hexdigest()
        result.append(dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
            device=str(tensor.device), storage_offset=tensor.storage_offset(), pointer=tensor.data_ptr(),
            backing_valid_bytes=nbytes, backing_sha256=value, version=tensor._version,
            is_neg=tensor.is_neg(), is_conj=tensor.is_conj()))
    return result


def _bit_equal(torch, left, right):
    return left.dtype == right.dtype and left.shape == right.shape and bool(torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8)))


def _compiled_cache(operator):
    """Read-only actual JIT cache inventory, specific to pinned Triton3.3.1."""
    result = []
    for kernel in operator._kernels or ():
        for device, entry in kernel.device_caches.items():
            require(type(entry) is tuple and len(entry) == 4 and type(entry[0]) is dict,
                    'Unexpected pinned Triton JIT cache schema')
            result.append(dict(kernel=kernel.__name__, device=int(device), compiled_entries=len(entry[0]),
                key_sha256s=sorted(hashlib.sha256(str(key).encode()).hexdigest() for key in entry[0])))
    return result


def case_status(case, *, finite, unchanged, output_contract=True, candidate_known=None, reference_known=None):
    mandatory = output_contract and finite and unchanged and (not case['exact'] or candidate_known is True)
    return dict(status=('CANDIDATE_ANALYTIC_CASE_PASSED' if case['exact'] else 'DIAGNOSTIC_COMPLETE') if mandatory else 'FAIL',
        mandatory_checks_passed=bool(mandatory), random_equivalence_gate='NOT_DEFINED',
        reference_analytic_gate='DIAGNOSTIC_ONLY', reference_equivalence_gate='NOT_ESTABLISHED_DIAGNOSTIC_ONLY',
        quality_gate='NOT_RUN', performance_gate='NOT_RUN', model_gate='NOT_RUN', token_gate='NOT_RUN')


def execute(output, cache, pins):
    output.mkdir(mode=0o700, exist_ok=False)
    write_json(output/'PLAN.json', plan())
    receipt = dict(status='FAIL', passed=False, actual_cuda=False, numerical_pass=False, candidate_analytic_smoke_passed=False,
        passed_scope='CANDIDATE_ANALYTIC_ONLY', reference_equivalence_gate='NOT_ESTABLISHED_DIAGNOSTIC_ONLY',
        quality_gate='NOT_RUN', performance_gate='NOT_RUN', model_gate='NOT_RUN', token_gate='NOT_RUN',
        speed_measured=False, bitwise_equivalence_claim=False, source_pins=pins, completed_cases=0,
        attempted_cases=0, reference_returned_cases=0, candidate_returned_cases=0, synchronized_cases=0,
        cuda_available=False, torch_cuda_initialized=False, current_case=None, current_stage='runtime_import',
        actual_cuda_definition=plan()['actual_cuda_definition'],
        expected_torch_version=TORCH_VERSION, expected_torch_cuda_version=TORCH_CUDA_VERSION,
        expected_triton_version=TRITON_VERSION)
    try:
        torch = importlib.import_module('torch')
        triton = importlib.import_module('triton')
        require(str(torch.__version__) == TORCH_VERSION and torch.version.cuda == TORCH_CUDA_VERSION
                and triton.__version__ == TRITON_VERSION, 'Pinned Torch2.7.1+cu128/CUDA12.8/Triton3.3.1 required')
        receipt['current_stage'] = 'gpu_identity_check'
        receipt['cuda_available'] = bool(torch.cuda.is_available())
        require(receipt['cuda_available'] and torch.cuda.device_count() == 1, 'Exactly one actual CUDA GPU required')
        properties = torch.cuda.get_device_properties(0)
        observed_uuid = _canonical_uuid(getattr(properties, 'uuid', ''))
        require(observed_uuid == GPU_UUID and properties.name.startswith('NVIDIA H100'), 'Fixed H100 identity mismatch')
        receipt.update(torch_cuda_initialized=bool(torch.cuda.is_initialized()), gpu_uuid=observed_uuid, gpu_name=properties.name,
            torch_version=str(torch.__version__), torch_cuda_version=torch.version.cuda, triton_version=triton.__version__,
            num_sms=properties.multi_processor_count, python_version=sys.version,
            environment={name: os.environ.get(name) for name in ('CUDA_VISIBLE_DEVICES', 'TRITON_CACHE_DIR', 'CUDA_CACHE_DISABLE')})
        receipt['current_stage'] = 'configure_fp32'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(torch.get_default_dtype() == torch.float32, 'Default FP32 dtype required')
        functional = importlib.import_module('torch.nn.functional')
        attention = importlib.import_module('torch.nn.attention')
        sys.path.insert(0, str(ROOT))
        module = importlib.import_module('exp.round_overhead.fp32_splitkv_candidate_v1')
        require(Path(module.__file__).resolve() == CANDIDATE and source_pins() == pins, 'Loaded candidate/source identity changed')
        operator = module.build_candidate(functional.scaled_dot_product_attention, enabled=True, cache_dir=cache)
        receipt.update(tf32=False, dtype='float32', reference_backend='EFFICIENT_ATTENTION only',
            reference_python_source_sha256=sha(Path(functional.__file__)), attention_context_source_sha256=sha(Path(attention.__file__)))
        with (output/'CASES.jsonl').open('x') as stream, torch.no_grad():
            for case in cases():
                receipt.update(current_case=case['case_id'], current_stage='construct_inputs',
                               attempted_cases=receipt['attempted_cases']+1)
                query, key, value, mask, known = _inputs(torch, case)
                receipt['current_stage'] = 'snapshot_inputs_before'
                before = _snapshot(torch, (query, key, value, mask))
                receipt['current_stage'] = 'reference_forward'
                with attention.sdpa_kernel(backends=[attention.SDPBackend.EFFICIENT_ATTENTION]):
                    reference = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                        dropout_p=0.0, is_causal=False)
                receipt['reference_returned_cases'] += 1
                receipt['current_stage'] = 'candidate_forward'
                actual = operator(query, key, value, mask, dropout_p=0.0, is_causal=False)
                receipt['candidate_returned_cases'] += 1
                receipt['current_stage'] = 'cuda_synchronize'
                torch.cuda.synchronize(0)
                receipt['synchronized_cases'] += 1
                receipt['actual_cuda'] = True
                receipt['current_stage'] = 'snapshot_inputs_after'
                after = _snapshot(torch, (query, key, value, mask))
                receipt['current_stage'] = 'diagnose_and_record'
                output_contract = all(type(tensor) is torch.Tensor and tensor.dtype == torch.float32
                    and tensor.shape == query.shape and tensor.device == query.device for tensor in (actual, reference))
                finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
                diagnostics = None
                if finite and output_contract:
                    delta = actual-reference
                    diagnostics = dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                        max_relative_floor1e6=float((delta.abs()/reference.abs().clamp_min(1e-6)).max()),
                        candidate_reference_bitwise=_bit_equal(torch, actual, reference))
                candidate_known = _bit_equal(torch, actual, known) if known is not None else None
                reference_known = _bit_equal(torch, reference, known) if known is not None else None
                output_hashes = {name: hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                    for name, tensor in (('candidate', actual), ('reference', reference))}
                row = dict(case, finite=finite, output_contract=output_contract, output_logical_bytes_sha256=output_hashes,
                    inputs_before=before, inputs_after=after,
                    errors_diagnostic_only=diagnostics, candidate_known_bitwise=candidate_known,
                    reference_known_bitwise=reference_known, compiled_cache=_compiled_cache(operator),
                    runtime_specials=dict(N=case['n'], non_specialized=list(module.DYNAMIC_STAGE_ONE),
                        q_strides=list(query.stride()), k_strides=list(key.stride()), v_strides=list(value.stride()),
                        mask_strides=None if mask is None else list(mask.stride())),
                    **case_status(case, finite=finite, unchanged=before == after, output_contract=output_contract,
                                  candidate_known=candidate_known, reference_known=reference_known))
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                receipt['completed_cases'] += 1
                require(row['mandatory_checks_passed'], f'Mandatory case checks failed: {case["case_id"]}')
        receipt.update(current_case=None, current_stage='final_source_check')
        require(receipt['completed_cases'] == receipt['attempted_cases'] == receipt['synchronized_cases'] ==
                receipt['reference_returned_cases'] == receipt['candidate_returned_cases'] == 112
                and source_pins() == pins, 'Incomplete or source-changed operator gate')
        receipt.update(status='CANDIDATE_ANALYTIC_SMOKE_PASSED', passed=True, candidate_analytic_smoke_passed=True,
            random_equivalence_gate='NOT_DEFINED_DIAGNOSTIC_ONLY', exact_simple_cases=16, random_diagnostic_cases=96,
            compiled_cache=_compiled_cache(operator), cases_sha256=sha(output/'CASES.jsonl'), current_stage='complete')
    except Exception as error:
        receipt.update(status='FAIL', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    write_json(output/'TERMINAL.json', receipt)
    return receipt


def fresh_process():
    return not any(name == 'torch' or name.startswith('torch.') or name == 'triton' or name.startswith('triton.')
                   for name in sys.modules)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-cuda', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.execute_cuda:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    try:
        require(fresh_process(), 'A fresh owned subprocess without preloaded Torch/Triton is mandatory')
        pins = source_pins()
        output, cache = preflight_paths(args.output, args.cache_dir)
        receipt = execute(output, cache, pins)
    except Exception as error:
        print(f'EXECUTION_REFUSED: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    print(json.dumps({key: receipt[key] for key in ('status', 'passed', 'actual_cuda', 'completed_cases')}, sort_keys=True))
    return 0 if receipt['candidate_analytic_smoke_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())


