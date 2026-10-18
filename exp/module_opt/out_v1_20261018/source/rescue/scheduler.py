from concurrent.futures import Future, ThreadPoolExecutor
import time

import torch

from . import features
from .messages import LengthPacket, ProtocolError
from .policy import choose
from .transport import LengthClient


class RoundJob:
    def __init__(self, context, future, condition=None):
        self.context, self.future = context, future
        self.condition = condition

    def publish(self, packet):
        self.context.set_result(packet)

    def abort(self, error):
        if not self.context.done():
            self.context.set_exception(error)
        if self.condition is not None and not self.condition.done():
            self.condition.set_exception(error)
        if hasattr(self, 'compute') and not self.future.done():
            self.future.set_exception(error)


class Scheduler:
    def __init__(self, config, model, cost, device):
        self.config, self.model, self.cost, self.device = config, model, cost, torch.device(device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rescue-planner') if config.asynchronous else None
        self.stream = torch.cuda.Stream(device=device) if config.asynchronous and self.device.type == 'cuda' else None
        self.client = LengthClient(config.length_endpoint, config.rpc_timeout) if config.length_endpoint else None
        self.pending = None
        self.featureless = not getattr(model, 'requires_features', True)
        self.conditioned = not self.featureless and getattr(model, 'group', None) in ('attention_injected','attention_injected_light') and config.length_policy == 'risk'
        if self.conditioned and self.client is not None:
            raise ValueError('Injected 36-feature policy currently requires local inference')

    def begin(self, t, s, progress, remaining):
        if self.config.length_policy == 'fixed':
            return None
        if self.pending is not None and not self.pending.future.done():
            raise RuntimeError('Bounded scheduler still owns a previous round')
        snapshot_ready = None
        if self.stream is not None:
            snapshot_ready = torch.cuda.Event()
            snapshot_ready.record(torch.cuda.current_stream(self.device))
        context = Future()
        condition = Future() if self.conditioned else None

        def conditioned_features(base, packet):
            if self.featureless:
                return torch.zeros(36, device=self.device)
            vector = features.finish(base, t, s, packet.positions,include_read=getattr(self.model,'requires_read',True))
            if condition is not None:
                extra, ready = condition.result()
                if self.stream is not None and ready is not None:
                    self.stream.wait_event(ready)
                vector[3] = packet.refresh_age/self.config.refresh_max
                vector = torch.cat((vector, extra))
            return vector

        @torch.no_grad()
        def compute():
            if self.stream is not None:
                with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                    self.stream.wait_event(snapshot_ready)
                    base = None if self.featureless else features.prepare(t, s, progress, remaining, self.config.recent)
                    packet = context.result()
                    if packet.ready is not None:
                        self.stream.wait_event(packet.ready)
                    vector = conditioned_features(base, packet)
                    return self._decision(packet.key, vector, remaining)
            base = None if self.featureless else features.prepare(t, s, progress, remaining, self.config.recent)
            packet = context.result()
            return self._decision(packet.key, conditioned_features(base, packet), remaining)

        if self.executor is None:
            # R-independent features run once, before M1. No thread/event tax.
            base = None if self.featureless else features.prepare(t, s, progress, remaining, self.config.recent)
            job = RoundJob(context, Future(), condition)
            job.compute = lambda packet: self._decision(packet.key, conditioned_features(base, packet), remaining)
        else:
            job = RoundJob(context, self.executor.submit(compute), condition)
        self.pending = job
        return job

    def publish(self, job, packet, remaining):
        if job is None:
            return LengthPacket(packet.key, min(self.config.fixed_k, remaining), 0.)
        job.publish(packet)
        job.packet = packet
        if self.conditioned:
            return None
        if self.executor is None:
            try:
                decision = job.compute(packet)
            except BaseException as error:
                job.future.set_exception(error)
                raise
            job.future.set_result(decision)
            return decision
        return None

    def condition(self, job, extra):
        if not self.conditioned:
            return
        ready = None
        if self.stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        job.condition.set_result((extra, ready))
        if self.executor is None:
            try:
                job.future.set_result(job.compute(job.packet))
            except BaseException as error:
                job.future.set_exception(error)
                raise

    def _decision(self, key, vector, remaining):
        if self.client is not None:
            try:
                return self.client.choose(key, vector, self.cost, remaining, self.config.max_draft)
            except (OSError, EOFError, ProtocolError) as error:
                return LengthPacket(key, min(self.config.fallback_k, remaining), 0., type(error).__name__)
        cap, score = choose(self.model, vector, self.cost, key.prefix, remaining, self.config.max_draft)
        return LengthPacket(key, cap, score)

    def resolve(self, job, key, remaining):
        start = time.perf_counter()
        ready = job.future.done()
        packet = job.future.result().validate(key, min(remaining, self.config.max_draft))
        return packet, ready, time.perf_counter()-start

    def drain(self):
        if self.pending is not None:
            try:
                self.pending.future.result()
            finally:
                self.pending = None

    def close(self):
        try:
            self.drain()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
            if self.client is not None:
                self.client.close()
