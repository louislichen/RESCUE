"""Opt-in host/stream intervals; stream elapsed includes queueing and idle time."""
import time
from collections import defaultdict
import torch


class Timeline:
    def __init__(self, device, enabled=False):
        self.device, self.enabled = torch.device(device), enabled
        self.intervals = []

    def begin(self, name):
        if not self.enabled:
            return None
        event = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        if event is not None:
            event.record(torch.cuda.current_stream(self.device))
        return name, time.perf_counter(), event

    def end(self, start):
        if start is None:
            return
        name, tick, event = start
        end = torch.cuda.Event(enable_timing=True) if event is not None else None
        if end is not None:
            end.record(torch.cuda.current_stream(self.device))
        self.intervals.append((name, time.perf_counter()-tick, event, end))

    def result(self):
        # The caller synchronizes only once, at request completion.
        groups = defaultdict(lambda: dict(calls=0, host_seconds=0., stream_seconds=0.))
        for name, wall, start, end in self.intervals:
            group = groups[name]
            group['calls'] += 1
            group['host_seconds'] += wall
            if start is not None:
                group['stream_seconds'] += start.elapsed_time(end)/1000
        return dict(stages=dict(groups), scope='Instrumentation only: overlapping stream/host intervals cannot be added as independent kernel costs')
