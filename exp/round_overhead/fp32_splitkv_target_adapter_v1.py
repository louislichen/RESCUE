"""Default-off, scoped Target-only adapter for an unverified FP32 operator.

No engine/source/module global is patched. Enabled entry validates the fixed
Qwen2 Target and existing efficient forward bytecode, then binds 28 private
FunctionType clones whose private globals contain an F proxy. Every original
bound method is restored on exit, including failed construction or execution.
Draft is never accepted by this API or traversed. Both comparison arms may use
the same scope separately with the same Target, after the prior scope closes.

Only CUDA FP32 folded [4,7,Q2..5,128], noncausal calls dispatch the existing
candidate. Q1, prefill, BF16 and unsupported metadata continue through the
original FSDPA. Candidate exceptions propagate, never silently fall back.
No-grad and both TF32 flags off are mandatory throughout an enabled scope.
Counters contain CPU metadata only, with no tensor readback/synchronization.

This is CPU lifecycle preparation, NOT CUDA/model/Target-Token/speed proof.
"""
import hashlib
import importlib
from pathlib import Path
import types


EFFICIENT_SOURCE_SHA256 = 'baf97e3c5fc8d9ff5e6f91c5de7be5198b550da82da29a553781584bf679a108'
CANDIDATE_SOURCE_SHA256 = '9f133cc824915afd09d9ad25a1b16e6cd721522e1f583cabd5e102235358cfd2'
ROOT = Path(__file__).resolve().parents[2]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _runtime_contract(torch):
    _require(not torch.is_grad_enabled(), 'Enabled Target scope requires no_grad')
    _require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
             'Enabled Target scope requires both TF32 flags already off')


def _reference_forward_code(path, filename):
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == EFFICIENT_SOURCE_SHA256,
             'Fixed efficient_attention source SHA mismatch')
    top = compile(data, filename, 'exec', dont_inherit=True)
    installers = [c for c in top.co_consts if isinstance(c, types.CodeType) and c.co_name == 'install_efficient_attention']
    _require(len(installers) == 1, 'Fixed efficient installer code missing')
    forwards = [c for c in installers[0].co_consts if isinstance(c, types.CodeType) and c.co_name == 'forward']
    _require(len(forwards) == 1, 'Fixed efficient bound forward code missing')
    return forwards[0]


def _validate_target(model, torch, functional):
    config = model.config
    _require(config.model_type == 'qwen2' and config.num_hidden_layers == 28
             and config.num_attention_heads == 28 and config.num_key_value_heads == 4
             and config.hidden_size == 3584 and getattr(config, 'head_dim', 128) == 128,
             'Scope requires fixed Qwen2 Target: 28 layers, 28Q/4KV heads, head128')
    layers = tuple(model.model.layers)
    _require(len(layers) == 28 and len({id(layer.self_attn) for layer in layers}) == 28,
             'Target must contain exactly28 distinct attention modules')
    originals, compiled = [], {}
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        _require(getattr(attention, '_rescue_efficient', False) is True
                 and attention.layer_idx == index and attention.num_heads == 28
                 and attention.num_key_value_heads == 4 and attention.num_key_value_groups == 7
                 and attention.head_dim == 128 and attention.hidden_size == 3584,
                 f'Layer{index} is not the fixed installed Target attention')
        bound = attention.forward
        _require(isinstance(bound, types.MethodType) and bound.__self__ is attention
                 and isinstance(bound.__func__, types.FunctionType), f'Layer{index} needs installed bound forward')
        source = bound.__func__
        _require(source.__name__ == 'forward' and source.__qualname__ == 'install_efficient_attention.<locals>.forward',
                 f'Layer{index} forward provenance mismatch')
        filename = source.__code__.co_filename
        path = Path(filename).resolve(strict=True)
        if filename not in compiled:
            compiled[filename] = _reference_forward_code(path, filename)
        _require(source.__code__ == compiled[filename], f'Layer{index} efficient forward bytecode mismatch')
        _require(source.__globals__.get('torch') is torch and source.__globals__.get('F') is functional
                 and isinstance(functional, types.ModuleType) and functional.__name__ == 'torch.nn.functional',
                 f'Layer{index} needs original Torch/F module globals; nested scopes are unsupported')
        originals.append((attention, bound))
    return originals


def _dispatch_reason(torch, args, kwargs):
    """Inspect shapes/dtypes/devices only; never inspect a Tensor's values."""
    if len(args) != 3 or any(k not in ('attn_mask', 'dropout_p', 'is_causal', 'scale', 'enable_gqa') for k in kwargs):
        return 'unsupported_call_signature', None, None
    query, key, value = args
    if any(type(t) is not torch.Tensor for t in (query, key, value)):
        return 'non_plain_tensor', None, None
    shape = tuple(query.shape)
    q = int(shape[-2]) if len(shape) >= 2 else None
    n = int(key.shape[-2]) if len(key.shape) >= 2 else None
    if any(t.dtype != torch.float32 for t in (query, key, value)):
        return 'non_fp32', q, n
    if any(t.device.type != 'cuda' for t in (query, key, value)):
        return 'non_cuda', q, n
    if len(shape) != 4 or shape[:2] != (4, 7) or shape[-1] != 128:
        return 'non_folded_query', q, n
    if q == 1:
        return 'query_length_1', q, n
    if q is None or not 2 <= q <= 5:
        return 'query_length_out_of_scope', q, n
    if kwargs.get('is_causal', False) is not False:
        return 'causal', q, n
    if kwargs.get('dropout_p', 0.) != 0 or kwargs.get('scale') is not None or kwargs.get('enable_gqa', False) is not False:
        return 'unsupported_attention_options', q, n
    if n is None or not q <= n <= 65536 or tuple(key.shape) != (4, 7, n, 128) or tuple(value.shape) != tuple(key.shape):
        return 'key_value_shape_out_of_scope', q, n
    return None, q, n


class _FunctionalProxy:
    def __init__(self, original, candidate, torch, layer, records):
        self._original, self._candidate, self._torch = original, candidate, torch
        self._layer, self._records = layer, records

    def __getattr__(self, name):
        return getattr(self._original, name)

    def scaled_dot_product_attention(self, *args, **kwargs):
        _runtime_contract(self._torch)
        reason, queries, keys = _dispatch_reason(self._torch, args, kwargs)
        row = dict(layer=self._layer, query_tokens=queries, key_tokens=keys,
                   action='original' if reason else 'candidate', fallback_reason=reason, outcome='started')
        self._records.append(row)
        operation = self._original.scaled_dot_product_attention if reason else self._candidate
        try:
            result = operation(*args, **kwargs)
        except BaseException as error:
            row.update(outcome='error', error_type=type(error).__name__)
            raise
        row['outcome'] = 'returned'
        return result


def _clone_forward(source, proxy):
    private = types.FunctionType(source.__code__, dict(source.__globals__, F=proxy), source.__name__,
                                 source.__defaults__, source.__closure__)
    private.__kwdefaults__ = source.__kwdefaults__
    private.__annotations__ = dict(source.__annotations__)
    private.__dict__.update(source.__dict__)
    private.__doc__, private.__module__, private.__qualname__ = source.__doc__, source.__module__, source.__qualname__
    return private


def _dependencies():
    torch = importlib.import_module('torch')
    functional = importlib.import_module('torch.nn.functional')
    candidate = importlib.import_module('exp.round_overhead.fp32_splitkv_candidate_v1')
    path = ROOT/'exp/round_overhead/fp32_splitkv_candidate_v1.py'
    _require(Path(candidate.__file__).resolve() == path
             and hashlib.sha256(path.read_bytes()).hexdigest() == CANDIDATE_SOURCE_SHA256,
             'Fixed candidate source provenance mismatch')
    return torch, functional, candidate.build_candidate


class TargetAttentionScope:
    """Use ``with TargetAttentionScope(target, enabled=True, cache_dir=...)``.

    Default-off construction/entry does not import Torch/Triton, access model
    attributes, validate/create a cache, allocate tensors or wrap any method.
    Factory/runtime injection is intentionally absent from this public API.
    The private _activate helper is dependency-injected only for CPU proofs.
    """
    def __init__(self, model, *, enabled=False, cache_dir=None):
        _require(type(enabled) is bool, 'Boolean enabled required')
        self.model, self.enabled, self.cache_dir = model, enabled, cache_dir
        self.state, self._installed, self._snapshots, self._records = 'new', [], [], []

    def __enter__(self):
        _require(self.state == 'new', 'Scope is one-shot; nested/reused scopes unsupported')
        if not self.enabled:
            self.state = 'disabled'
            return self.model
        try:
            return self._activate(*_dependencies())
        except BaseException:
            self._restore()
            self.state = 'failed'
            raise

    def _activate(self, torch, functional, factory):
        """Private only: tests supply metadata-only fakes, never public opt-outs."""
        _require(self.state == 'new' and self.enabled is True, 'Fresh enabled scope required')
        _runtime_contract(torch)
        originals = _validate_target(self.model, torch, functional)
        candidate = factory(functional.scaled_dot_product_attention, enabled=True, cache_dir=self.cache_dir)
        _require(callable(candidate), 'Candidate factory must return a callable')
        try:
            for layer, (attention, original) in enumerate(originals):
                source = original.__func__
                self._snapshots.append((source, dict(source.__globals__), source.__defaults__, source.__closure__,
                                        source.__kwdefaults__, dict(source.__annotations__)))
                proxy = _FunctionalProxy(functional, candidate, torch, layer, self._records)
                private = types.MethodType(_clone_forward(source, proxy), attention)
                # Record before setattr: custom modules may mutate then raise.
                self._installed.append((attention, original))
                attention.forward = private
        except BaseException:
            self._restore()
            self.state = 'failed'
            raise
        self.state = 'active'
        return self.model

    def _restore(self):
        problems = []
        for attention, original in reversed(self._installed):
            try:
                attention.forward = original
                if attention.forward is not original:
                    problems.append('original bound method identity was not restored')
            except BaseException as error:
                problems.append('restore failed: '+type(error).__name__)
        self._installed.clear()
        for source, original_globals, defaults, closure, kwdefaults, annotations in self._snapshots:
            if (source.__globals__.keys() != original_globals.keys()
                    or any(source.__globals__[key] is not val for key, val in original_globals.items())
                    or source.__defaults__ is not defaults or source.__closure__ is not closure
                    or source.__kwdefaults__ is not kwdefaults or source.__annotations__ != annotations):
                problems.append('original source globals/defaults/closure/annotations changed')
        self._snapshots.clear()
        if problems:
            raise RuntimeError('; '.join(problems))

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._restore()
        finally:
            self.state = 'closed'
        return False

    def summary(self):
        return dict(state=self.state, enabled=self.enabled, cpu_metadata_only=True,
            tensor_readbacks=0, tensor_allocations_by_adapter=0,
            cuda_or_model_validated=False, token_gate='NOT_RUN', performance_gate='NOT_RUN',
            calls=len(self._records), candidate_calls=sum(r['action'] == 'candidate' for r in self._records),
            original_calls=sum(r['action'] == 'original' for r in self._records),
            records=[dict(row) for row in self._records])

    def stats(self):
        """CPU metadata snapshot; callers may difference counts/slice records."""
        return self.summary()
