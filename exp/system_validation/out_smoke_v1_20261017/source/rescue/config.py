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

    def __post_init__(self):
        for name in ('budget', 'recent', 'chunk_size', 'coverage_candidates', 'coverage_batch',
                     'refresh_every', 'fixed_k', 'max_draft', 'max_new_tokens', 'fallback_k'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.budget <= self.recent or self.max_draft > 12 or self.fixed_k > self.max_draft or self.fallback_k > self.max_draft:
            raise ValueError('Invalid recent/budget or length/horizon bounds')
        if type(self.inject_tokens) is not int or self.inject_tokens < 0:
            raise ValueError('inject_tokens must be nonnegative')
        if self.selector not in ('teacher', 'joint', 'coverage', 'read_joint', 'full'):
            raise ValueError('Unknown context selector')
        if self.selector == 'read_joint' and self.chunk_size != 1:
            raise ValueError('Read-preservation compatibility selector uses token granularity')
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
