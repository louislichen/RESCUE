"""Isolated, default-off FP32 split-KV operator prototype; NEVER an engine patch.

No CUDA compilation/numerical/equivalence/speed proof exists for this candidate.
It changes floating-point reduction order and does not promise bitwise equality.
Only a future separately authorized owned-cache CUDA gate may execute it.
That gate must use a fresh owned subprocess. Metadata guards do not secure an
arbitrarily monkeypatched interpreter or provide a process-isolation sandbox.

Interface: build_candidate(fallback, enabled=False, cache_dir=None). Disabled
construction returns the original callable identity before importing Torch or
Triton. Enabled calls require CUDA FP32 folded [4,7,Q,128] tensors, Q=2..5,
dropout=0, is_causal=False, and an optional broadcast rank-4 float32/bool mask.
Finite inputs and finite-or-minus-infinity masks are the intended numeric scope;
nonfinite propagation and any Target-token property remain unverified.

The two GPU kernels use SPLITS=8 and BLOCK_K=128 KEYS, not 128K tokens. All
floating loads/products/reductions/accumulators/stores use FP32; no tl.dot is
used. N and ALL potentially N-dependent strides are runtime, non-specialized
arguments. N is explicitly bounded to 65,536 keys (including appended tokens),
not arbitrary int32 lengths. CPU checks do NOT prove absence of recompilation.

Official interfaces inspected (Triton 3.3.1, matching Torch2.7.1's generation):
https://triton-lang.org/main/python-api/generated/triton.jit.html
https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/runtime/jit.py
https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/runtime/cache.py
https://triton-lang.org/main/python-api/generated/triton.language.sum.html
"""
from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import stat
import types


SPLITS = 8
BLOCK_K = 128
HEAD_DIM = 128
FOLDED_BATCH = 4
FOLDED_HEADS = 7
MAX_KEYS = 65536
DYNAMIC_STAGE_ONE = (
    'N', 'Q0', 'Q1', 'Q2', 'K0', 'K1', 'K2', 'V0', 'V1', 'V2',
    'M0', 'M1', 'M2', 'M3',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def split_key_ranges(n):
    """CPU metadata only: eight contiguous key ranges used by stage one."""
    _require(type(n) is int and 1 <= n <= MAX_KEYS, 'N must lie in the prototype scope1..65536 keys')
    tiles_per_split = ((n + BLOCK_K - 1)//BLOCK_K + SPLITS - 1)//SPLITS
    return tuple((min(s*tiles_per_split*BLOCK_K, n),
                  min((s+1)*tiles_per_split*BLOCK_K, n)) for s in range(SPLITS))


def _contract(query, key, value, attn_mask, dropout_p, is_causal, scale,
              enable_gqa, *, require_cuda=True):
    """Shape/stride metadata validation only; never evaluates tensor values.

    require_cuda=False exists solely for CPU contract tests. The enabled public
    callable does not expose this switch and always requires actual CUDA.
    """
    torch = importlib.import_module('torch')
    _require(type(is_causal) is bool and not is_causal, 'is_causal must be False')
    _require(type(enable_gqa) is bool and not enable_gqa, 'Inputs are already folded; enable_gqa must be False')
    _require(type(dropout_p) in (int, float) and dropout_p == 0, 'Only dropout_p=0 is supported')
    _require(scale is None, 'Custom attention scale is unsupported')
    for name, tensor in (('query', query), ('key', key), ('value', value)):
        _require(type(tensor) is torch.Tensor, f'{name} must be a plain Tensor')
        _require(tensor.layout == torch.strided and tensor.ndim == 4, f'{name} must be strided rank4')
        _require(tensor.dtype == torch.float32, f'{name} must be float32')
        _require(not tensor.is_neg() and not tensor.is_conj(),
                 f'{name} cannot have unresolved negative/conjugate view bits')
        _require(not tensor.requires_grad, 'Autograd is unsupported')
        _require(tensor.stride(-1) == 1 and all(s >= 0 for s in tensor.stride()),
                 f'{name} requires contiguous head dimension and nonnegative strides')
        _require(sum((size-1)*stride for size, stride in zip(tensor.shape, tensor.stride())) < 2**31,
                 f'{name} exceeds the prototype int32 offset scope')
        if require_cuda:
            _require(tensor.device.type == 'cuda', 'Enabled candidate requires CUDA; no CPU fallback')
    _require(query.device == key.device == value.device, 'All inputs must share one device')
    _require(query.shape[0:2] == (FOLDED_BATCH, FOLDED_HEADS) and query.shape[-1] == HEAD_DIM,
             'query shape must be [4,7,Q,128]')
    queries = query.shape[2]
    _require(2 <= queries <= 5, 'Only short query lengths2..5 are supported')
    n = key.shape[2]
    _require(queries <= n <= MAX_KEYS and key.shape == value.shape and
             key.shape[0:2] == (FOLDED_BATCH, FOLDED_HEADS) and key.shape[-1] == HEAD_DIM,
             'K/V shapes must match [4,7,N,128], with Q<=N<=65536 keys')
    mask_strides, mask_kind = (0, 0, 0, 0), 0
    if attn_mask is not None:
        _require(type(attn_mask) is torch.Tensor and attn_mask.layout == torch.strided and attn_mask.ndim == 4,
                 'Mask must be a plain strided rank4 Tensor')
        _require(attn_mask.dtype in (torch.float32, torch.bool), 'Mask must be float32 additive or bool')
        _require(not attn_mask.is_neg() and not attn_mask.is_conj(),
                 'Mask cannot have unresolved negative/conjugate view bits')
        _require(attn_mask.device == query.device and not attn_mask.requires_grad,
                 'Mask must share the input device and require no gradients')
        _require(all(size in (1, extent) for size, extent in zip(attn_mask.shape, (4, 7, queries, n))),
                 'Mask must broadcast to [4,7,Q,N]')
        _require(all(s >= 0 for s in attn_mask.stride()), 'Negative mask strides are unsupported')
        _require(sum((size-1)*stride for size, stride in zip(attn_mask.shape, attn_mask.stride())) < 2**31,
                 'Mask exceeds the prototype int32 offset scope')
        mask_strides = tuple(0 if size == 1 else stride for size, stride in zip(attn_mask.shape, attn_mask.stride()))
        mask_kind = 2 if attn_mask.dtype == torch.bool else 1
    return dict(n=n, queries=queries, query_strides=query.stride()[:3],
                key_strides=key.stride()[:3], value_strides=value.stride()[:3],
                mask_strides=mask_strides, mask_kind=mask_kind)


def _cache_identity(cache_dir):
    _require(cache_dir is not None, 'Explicit existing owned cache_dir is mandatory')
    path = Path(cache_dir)
    _require(path.is_absolute() and path.resolve(strict=True) == path,
             'Cache must be an absolute existing non-symlink canonical path')
    info = path.stat()
    _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o022,
             'Cache must be a user-owned directory without group/world write access')
    _require(path not in (Path('/'), Path.home().resolve()), 'Broad cache directory is forbidden')
    _require(os.environ.get('TRITON_CACHE_DIR') == str(path), 'TRITON_CACHE_DIR must already exactly name cache_dir')
    return path, (info.st_dev, info.st_ino, info.st_uid, info.st_mode)


def _check_compile_environment(cache_dir, identity):
    _require(_cache_identity(cache_dir)[1] == identity, 'Owned cache directory changed')
    # Triton3.3.1 checks `is not None`, so even an empty manager override is
    # different from absence and is not a supported default configuration.
    _require('TRITON_CACHE_MANAGER' not in os.environ, 'TRITON_CACHE_MANAGER must be absent, not empty')
    for name in ('TRITON_REMOTE_CACHE_BACKEND', 'TRITON_DUMP_DIR', 'TRITON_OVERRIDE_DIR'):
        _require(not os.environ.get(name), f'{name} override is unsupported')
    for name in ('TRITON_INTERPRET', 'TRITON_KERNEL_DUMP', 'TRITON_KERNEL_OVERRIDE'):
        _require(os.environ.get(name, '0') in ('', '0'), f'{name} must be disabled')
    # Do not let the CUDA driver place its own generated-code cache elsewhere.
    _require(os.environ.get('CUDA_CACHE_DISABLE') == '1', 'CUDA_CACHE_DISABLE=1 is required before future compilation')


def _stage_one(Q, K, V, MASK, PM, PL, PO,
               N, Q0, Q1, Q2, K0, K1, K2, V0, V1, V2, M0, M1, M2, M3,
               QUERIES: tl.constexpr, HEADS: tl.constexpr, D: tl.constexpr,
               SPLIT_COUNT: tl.constexpr, TILE_K: tl.constexpr, MASK_KIND: tl.constexpr,
               SCALE: tl.constexpr):
    # This ordinary Python function is NEVER called as Python. A private globals
    # copy receives tl only after explicit CUDA validation and lazy import.
    row, split = tl.program_id(0), tl.program_id(1)
    batch, head, query = row//(HEADS*QUERIES), (row//QUERIES) % HEADS, row % QUERIES
    dims = tl.arange(0, D)
    keys = tl.arange(0, TILE_K)
    q = tl.load(Q + batch*Q0 + head*Q1 + query*Q2 + dims).to(tl.float32)
    tiles_per_split = tl.cdiv(tl.cdiv(N, TILE_K), SPLIT_COUNT)
    offset = split*tiles_per_split*TILE_K
    stop = tl.minimum((split+1)*tiles_per_split*TILE_K, N)
    maximum = tl.full((), -float('inf'), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.full((D,), 0.0, tl.float32)
    while offset < stop:
        positions = offset + keys
        valid = positions < stop
        k = tl.load(K + batch*K0 + head*K1 + positions[:, None]*K2 + dims[None, :],
                    mask=valid[:, None], other=0.0).to(tl.float32)
        products = k*q[None, :]
        scores = tl.sum(products, axis=1, dtype=tl.float32)*SCALE
        if MASK_KIND != 0:
            mask_values = tl.load(MASK + batch*M0 + head*M1 + query*M2 + positions*M3,
                                  mask=valid, other=0)
            if MASK_KIND == 2:
                scores = tl.where(mask_values, scores, -float('inf'))
            else:
                scores = scores + mask_values.to(tl.float32)
        scores = tl.where(valid, scores, -float('inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        alpha = tl.where(new_maximum == -float('inf'), 0.0, tl.exp(maximum-new_maximum))
        probabilities = tl.where(new_maximum == -float('inf'), 0.0, tl.exp(scores-new_maximum))
        v = tl.load(V + batch*V0 + head*V1 + positions[:, None]*V2 + dims[None, :],
                    mask=valid[:, None], other=0.0).to(tl.float32)
        contribution = tl.sum(probabilities[:, None]*v, axis=0, dtype=tl.float32)
        numerator = numerator*alpha + contribution
        denominator = denominator*alpha + tl.sum(probabilities, axis=0, dtype=tl.float32)
        maximum = new_maximum
        offset += TILE_K
    partial = row*SPLIT_COUNT + split
    tl.store(PM + partial, maximum)
    tl.store(PL + partial, denominator)
    tl.store(PO + partial*D + dims, numerator)


def _stage_two(PM, PL, PO, OUT, D: tl.constexpr, SPLIT_COUNT: tl.constexpr):
    row = tl.program_id(0)
    splits, dims = tl.arange(0, SPLIT_COUNT), tl.arange(0, D)
    maximums = tl.load(PM + row*SPLIT_COUNT + splits).to(tl.float32)
    denominators = tl.load(PL + row*SPLIT_COUNT + splits).to(tl.float32)
    numerators = tl.load(PO + (row*SPLIT_COUNT + splits[:, None])*D + dims[None, :]).to(tl.float32)
    maximum = tl.max(maximums, axis=0)
    weights = tl.where(maximum == -float('inf'), 0.0, tl.exp(maximums-maximum))
    denominator = tl.sum(denominators*weights, axis=0, dtype=tl.float32)
    numerator = tl.sum(numerators*weights[:, None], axis=0, dtype=tl.float32)
    # Empty or fully masked splits have m=-inf,l=0,o=0; a fully masked row is 0.
    output = tl.where(denominator == 0.0, 0.0, numerator/denominator)
    tl.store(OUT + row*D + dims, output.to(tl.float32))


def _assert_default_cache_manager():
    """Reject sticky custom cache state without resetting any Triton global.

    In official Triton3.3.1 runtime/cache.py, get_cache_manager remembers both
    __cache_cls and __cache_cls_nme. Clearing TRITON_CACHE_MANAGER does not
    restore them. Missing/changed private schema is deliberately unsupported.
    This check is repeated before any later use, not just the first import.
    """
    cache = importlib.import_module('triton.runtime.cache')
    expected = getattr(cache, 'FileCacheManager', None)
    _require(expected is not None and getattr(cache, '__cache_cls', None) is expected
             and getattr(cache, '__cache_cls_nme', None) == 'DEFAULT',
             'Triton cache-manager state must still be the untouched default')


def _load_kernels():
    triton = importlib.import_module('triton')
    _assert_default_cache_manager()
    tl_module = importlib.import_module('triton.language')
    functions = []
    for source, dynamic in ((_stage_one, DYNAMIC_STAGE_ONE), (_stage_two, ())):
        private = types.FunctionType(source.__code__, dict(source.__globals__, tl=tl_module),
                                     source.__name__, source.__defaults__, source.__closure__)
        private.__annotations__ = dict(source.__annotations__)
        functions.append(triton.jit(do_not_specialize=dynamic,
                                   do_not_specialize_on_alignment=dynamic)(private))
    return tuple(functions)


class _Candidate:
    cuda_numerics_verified = False
    bitwise_equivalence_claim = False
    speed_verified = False
    engine_integrated = False

    def __init__(self, cache_dir, cache_identity):
        self.cache_dir, self.cache_identity = cache_dir, cache_identity
        self._kernels = None

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, *, scale=None, enable_gqa=False):
        meta = _contract(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        # CPU is rejected above, before cache checks/imports/allocations/fallback.
        _check_compile_environment(self.cache_dir, self.cache_identity)
        torch = importlib.import_module('torch')
        _require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                 'Future CUDA execution requires both TF32 flags already disabled')
        with torch.cuda.device(query.device):
            _require(not torch.cuda.is_current_stream_capturing(), 'This standalone candidate is not CUDA-Graph eligible')
            if self._kernels is None:
                self._kernels = _load_kernels()
            else:
                _assert_default_cache_manager()
            first, second = self._kernels
            rows = FOLDED_BATCH*FOLDED_HEADS*meta['queries']
            pm = torch.empty((rows, SPLITS), dtype=torch.float32, device=query.device)
            pl = torch.empty_like(pm)
            po = torch.empty((rows, SPLITS, HEAD_DIM), dtype=torch.float32, device=query.device)
            output = torch.empty(tuple(query.shape), dtype=torch.float32, device=query.device)
            first[(rows, SPLITS)](query, key, value, query if attn_mask is None else attn_mask,
                pm, pl, po, meta['n'], *meta['query_strides'], *meta['key_strides'],
                *meta['value_strides'], *meta['mask_strides'], QUERIES=meta['queries'],
                HEADS=FOLDED_HEADS, D=HEAD_DIM, SPLIT_COUNT=SPLITS, TILE_K=BLOCK_K,
                MASK_KIND=meta['mask_kind'], SCALE=1.0/math.sqrt(HEAD_DIM),
                num_warps=4, num_stages=1, enable_fp_fusion=False)
            second[(rows,)](pm, pl, po, output, D=HEAD_DIM, SPLIT_COUNT=SPLITS,
                            num_warps=4, num_stages=1, enable_fp_fusion=False)
        return output


def build_candidate(fallback, *, enabled=False, cache_dir=None):
    """No activation, imports, Tensor allocation or wrapping on the default path.

    Enabled construction validates an already-created owned cache and its
    already-set TRITON_CACHE_DIR. It never creates directories or changes env.
    Actual CUDA calls additionally require CUDA_CACHE_DISABLE=1 and no override,
    dump, interpreter or custom cache-manager settings. No fallback is used when
    enabled; unsupported input raises rather than silently changing execution.
    """
    _require(type(enabled) is bool and callable(fallback), 'Boolean enabled and callable fallback required')
    if not enabled:
        return fallback
    path, identity = _cache_identity(cache_dir)
    return _Candidate(path, identity)
