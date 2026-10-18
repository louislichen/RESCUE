from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SystemConfig:
    budget: int = 512
    recent: int = 64
    selector: str = 'coverage'
    chunk_size: int = 1
    student_weight: float = 1.
    coverage_candidates: int = 1024
    coverage_batch: int = 16
    refresh_every: int = 4
    inject_tokens: int = 128
    injection_strength: float = .5
    attention_backend: str = 'sdpa'
    schedule: str = 'auto'
    length_policy: str = 'risk'
    fixed_k: int = 4
    max_draft: int = 12
    max_new_tokens: int = 64
    length_endpoint: tuple | None = None
    rpc_timeout: float = 1.
    fallback_k: int = 1
    profile: bool = False
    prefill_chunk_size: int = 0
    optimized_execution: bool = True
    deferred_probes: bool = True
    memoized_gate_affine: bool = True
    working_mode: str = 'packed'
    injection_layers: tuple | None = None
    injection_rule: str = 'teacher'
    coverage_rows: int = 512
    coarse_chunk_size: int = 32
    refresh_policy: str = 'periodic'
    refresh_min: int = 1
    refresh_max: int = 16
    attention_drift: float = .15
    coverage_floor: float = .65
    prefill_precision: str = 'native'
    early_length_publish: bool = False
    coverage_anchor_fraction: float = 0.
    coverage_attention_weight: float = 0.

    def __post_init__(self):
        for name in ('optimized_execution', 'deferred_probes', 'memoized_gate_affine', 'early_length_publish'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if self.working_mode not in ('layerwise', 'packed', 'incremental'):
            raise ValueError('Unknown working-cache mode')
        if self.injection_rule not in ('teacher', 'joint'):
            raise ValueError('Unknown injection ranking')
        if self.injection_layers is not None:
            if (not isinstance(self.injection_layers, tuple) or
                any(type(v) is not int or v < 0 for v in self.injection_layers) or
                tuple(sorted(set(self.injection_layers))) != self.injection_layers):
                raise ValueError('Injection layers must be a sorted unique tuple')
        for name in ('budget', 'recent', 'chunk_size', 'coverage_candidates', 'coverage_batch',
                     'refresh_every', 'fixed_k', 'max_draft', 'max_new_tokens', 'fallback_k'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('coverage_rows', 'coarse_chunk_size', 'refresh_min', 'refresh_max'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.refresh_policy not in ('periodic', 'adaptive') or self.refresh_min > self.refresh_max:
            raise ValueError('Invalid refresh policy/bounds')
        if self.prefill_precision not in ('native', 'bfloat16'):
            raise ValueError('Invalid prefill precision')
        if not 0 <= self.attention_drift <= 1 or not 0 <= self.coverage_floor <= 1:
            raise ValueError('Drift/coverage thresholds must be probabilities')
        if (not 0 <= self.coverage_anchor_fraction <= 1 or
            not math.isfinite(self.coverage_attention_weight) or self.coverage_attention_weight < 0):
            raise ValueError('Invalid coverage attention anchors/weight')
        if self.budget <= self.recent or self.max_draft > 12 or self.fixed_k > self.max_draft or self.fallback_k > self.max_draft:
            raise ValueError('Invalid recent/budget or length/horizon bounds')
        if type(self.inject_tokens) is not int or self.inject_tokens < 0:
            raise ValueError('inject_tokens must be nonnegative')
        if type(self.prefill_chunk_size) is not int or self.prefill_chunk_size < 0:
            raise ValueError('prefill_chunk_size must be nonnegative')
        if self.selector not in ('teacher', 'joint', 'coverage', 'hierarchical', 'read_joint', 'full'):
            raise ValueError('Unknown context selector')
        if self.selector == 'read_joint' and self.chunk_size != 1:
            raise ValueError('Read-preservation compatibility selector uses token granularity')
        if self.selector == 'hierarchical' and self.chunk_size != 1:
            raise ValueError('Hierarchical selector uses coarse chunks and fine token outputs')
        if self.attention_backend not in ('eager', 'sdpa') or self.schedule not in ('serial', 'async', 'auto'):
            raise ValueError('Invalid attention backend or execution schedule')
        if self.length_policy not in ('fixed', 'risk'):
            raise ValueError('Unknown length policy')
        for name in ('student_weight', 'injection_strength', 'rpc_timeout'):
            v = getattr(self, name)
            if isinstance(v, bool) or not math.isfinite(v) or v < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.rpc_timeout <= 0:
            raise ValueError('rpc_timeout must be positive')
        if self.length_endpoint is not None:
            host, port = self.length_endpoint
            if not isinstance(host, str) or not host or type(port) is not int or not 0 < port < 65536:
                raise ValueError('Expected a valid length server host/port')

    @property
    def asynchronous(self):
        # Separate CPU/network execution is a reason to overlap. Do not force
        # two competing CUDA streams on a low-latency single GPU by default.
        return self.schedule == 'async' or self.schedule == 'auto' and self.length_endpoint is not None
