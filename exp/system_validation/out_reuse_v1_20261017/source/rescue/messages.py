from dataclasses import dataclass
import math


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class RoundKey:
    request: str
    round_index: int
    prefix: int
    workset_version: int

    def __post_init__(self):
        if not isinstance(self.request, str) or not self.request:
            raise ProtocolError('Missing request identity')
        if any(type(v) is not int or v < 0 for v in (self.round_index, self.prefix, self.workset_version)):
            raise ProtocolError('Invalid round/prefix/workset identity')

    def to_dict(self):
        return dict(request=self.request, round_index=self.round_index,
                    prefix=self.prefix, workset_version=self.workset_version)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'request', 'round_index', 'prefix', 'workset_version'}:
            raise ProtocolError('Invalid message identity fields')
        return cls(**value)


@dataclass(frozen=True)
class ContextPacket:
    key: RoundKey
    positions: object
    ready: object = None


@dataclass(frozen=True)
class LengthPacket:
    key: RoundKey
    cap: int
    score: float
    fallback: str | None = None

    def validate(self, expected, remaining):
        if self.key != expected:
            raise ProtocolError('Stale or mismatched length message')
        if type(self.cap) is not int or not 1 <= self.cap <= remaining or type(self.score) not in (int, float) or not math.isfinite(self.score):
            raise ProtocolError('Invalid length decision')
        return self
