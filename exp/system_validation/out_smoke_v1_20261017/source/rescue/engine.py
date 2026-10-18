"""Complete single-request Qwen2 greedy SD engine and its control pipeline."""
import time
import uuid

import torch

from .attention import DraftAttention, TargetObservation
from .cache import NativeCache, WorkingCache
from .injection import ContentInjector
from .messages import ContextPacket, RoundKey
from .scheduler import Scheduler
from .selection import ContextSelector, ReadState


class RescueEngine:
    def __init__(self, teacher, student, config, mapper=None, risk=None, cost=None, memoize=True, vocab_size=None):
        if teacher.config.model_type != 'qwen2' or student.config.model_type != 'qwen2':
            raise ValueError('This model adapter supports Qwen2; other architectures require an adapter')
        self.teacher, self.student, self.config = teacher, student, config
        self.device = teacher.model.embed_tokens.weight.device
        if student.model.embed_tokens.weight.device != self.device:
            raise ValueError('Model inference currently uses one device; length service may be remote')
        if config.length_policy == 'risk' and risk is None and config.length_endpoint is None:
            raise ValueError('Risk policy requires a checkpoint or remote length service')
        self.mapper, self.risk, self.memoize = mapper, risk, memoize
        self.vocab = vocab_size or min(teacher.config.vocab_size, student.config.vocab_size)
        if mapper is not None:
            expected = (teacher.config.num_key_value_heads*(teacher.config.hidden_size//teacher.config.num_attention_heads),
                        student.config.num_key_value_heads*(student.config.hidden_size//student.config.num_attention_heads),
                        student.config.hidden_size)
            actual = tuple(mapper.config[k] for k in ('source_dim', 'value_dim', 'query_dim'))
            if expected != actual or max(mapper.layers) >= len(student.model.layers) or max(mapper.teacher_layers) >= len(teacher.model.layers):
                raise ValueError('Mapper dimensions/layer sources do not match these base models')
        teacher.eval().requires_grad_(False)
        student.eval().requires_grad_(False)
        self.cost = cost or dict(verify_beta=[.02, 0., .0003, 0.], student_step=.01, cache_update=0.)
        self.observation = TargetObservation(teacher).install()
        self.scheduler = Scheduler(config, risk, self.cost, self.device)
        self.native_t = self.native_s = self.working = self.injector = None
        self.closed = False

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _buffers(self, capacity):
        if self.native_t is not None and self.native_t.capacity >= capacity:
            self.native_t.reset()
            self.native_s.reset()
            self.working.invalidate()
            self.injector.valid.zero_()
            self.injector.clear()
            self.injector.projected_positions = 0
            self.working.gathered_layers = 0
            return
        capacity = 1 << (capacity-1).bit_length()
        def native(model):
            c = model.config
            return NativeCache(c.num_hidden_layers, c.num_key_value_heads,
                               c.hidden_size//c.num_attention_heads, capacity,
                               model.model.embed_tokens.weight.dtype, self.device)
        self.native_t, self.native_s = native(self.teacher), native(self.student)
        self.working = WorkingCache(self.native_s, capacity if self.config.selector == 'full' else self.config.budget,
                                    self.config.max_draft)
        self.injector = ContentInjector(self.mapper, capacity, self.device,
                                       self.config.inject_tokens, self.config.injection_strength, self.memoize)

    def _ids(self, tokens):
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def _forward(self, model, tokens, cache):
        return model.model(input_ids=self._ids(tokens), past_key_values=cache,
                           use_cache=True, output_attentions=False, return_dict=True)

    @torch.no_grad()
    def reference(self, input_ids, eos_token_id=None):
        self.observation.mode = None
        self.synchronize()
        start = time.perf_counter()
        out = self._forward(self.teacher, input_ids[:-1], None)
        cache = out.past_key_values
        self.synchronize()
        prefill_end = time.perf_counter()
        current, generated = input_ids[-1], []
        for _ in range(self.config.max_new_tokens):
            out = self._forward(self.teacher, [current], cache)
            cache = out.past_key_values
            current = int(self.teacher.lm_head(out.last_hidden_state[:, -1:])[0, -1, :self.vocab].float().argmax())
            generated.append(current)
            if current == eos_token_id:
                break
        self.synchronize()
        end = time.perf_counter()
        return dict(generated=generated, prefill_seconds=prefill_end-start,
                    decode_seconds=end-prefill_end, total_seconds=end-start,
                    decode_tps=len(generated)/(end-prefill_end), total_tps=len(generated)/(end-start))

    @torch.no_grad()
    def generate(self, input_ids, eos_token_id=None):
        if self.closed:
            raise RuntimeError('Engine has been closed')
        if len(input_ids) < 3:
            raise ValueError('The attention snapshot requires at least three prompt tokens')
        c = self.config
        self.synchronize()
        request_start = time.perf_counter()
        self._buffers(len(input_ids)+c.max_new_tokens+c.max_draft+1)
        draft_attn = DraftAttention(self.student, self.working, self.injector, c.attention_backend).install()
        request_id = uuid.uuid4().hex
        selector = ContextSelector(c)
        trace, generated = [], []
        profile_steps, profile_verify, profile_overhead = [], [], []
        job = None
        proposed = accepted = 0
        try:
            self.observation.mode = 'last'
            self._forward(self.teacher, input_ids[:-1], self.native_t)
            self._forward(self.student, input_ids[:-1], self.native_s)
            tr, sr = self.observation.rows, draft_attn.shadow
            self.observation.rows = None
            draft_attn.prefill = False
            self.synchronize()
            prefill_end = time.perf_counter()
            current = input_ids[-1]
            while len(generated) < c.max_new_tokens:
                n = self.native_t.get_seq_length()
                if n != self.native_s.get_seq_length():
                    raise RuntimeError('Target/Draft committed prefixes diverged')
                progress = len(generated)
                remaining = c.max_new_tokens-progress
                t = ReadState(tr, self.native_t[-1][1][0])
                s = ReadState(sr, self.native_s[-1][1][0])
                # M3 begins its R-independent work while M1 selects positions.
                job = self.scheduler.begin(t, s, progress, remaining)
                selection_start = time.perf_counter()
                selected = selector.select(t, s, len(trace))
                key = RoundKey(request_id, len(trace), n, selector.version)
                event = None
                if self.scheduler.stream is not None:
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(self.device))
                context = ContextPacket(key, selected, event)
                decision = self.scheduler.publish(job, context, remaining)
                self.working.bind(self.native_s, selected)
                self.injector.bind(self.native_t, self.native_s, selected, t.a.mean(0), c.recent)
                overhead = time.perf_counter()-selection_start
                if c.profile:
                    profile_overhead.append(overhead)
                draft, probes = [], []
                next_input = current
                cap = decision.cap if decision is not None else 1
                ready, wait = None, 0.
                while len(draft) < cap:
                    tick = time.perf_counter()
                    logits = draft_attn.step(next_input, self.native_s)
                    next_input = int(logits[0, :self.vocab].argmax())
                    if c.profile:
                        profile_steps.append(time.perf_counter()-tick)
                    draft.append(next_input)
                    probes.append(draft_attn.shadow)
                    if decision is None:
                        decision, ready, wait = self.scheduler.resolve(job, key, remaining)
                        cap = decision.cap
                    if next_input == eos_token_id:
                        break
                self.scheduler.drain()
                job = None
                self.observation.mode = 'all'
                tick = time.perf_counter()
                out = self._forward(self.teacher, [current]+draft, self.native_t)
                checks = self.teacher.lm_head(out.last_hidden_state)[0, :, :self.vocab].float().argmax(-1).tolist()
                verify_seconds = time.perf_counter()-tick
                if c.profile:
                    profile_verify.append(dict(n=n, k=len(draft), seconds=verify_seconds))
                count = 0
                for index, token in enumerate(draft):
                    if token != checks[index]:
                        break
                    count += 1
                    if token == eos_token_id:
                        break
                proposed += len(draft)
                accepted += count
                generated.extend(draft[:count])
                stop = len(generated) >= c.max_new_tokens or (bool(generated) and generated[-1] == eos_token_id)
                if not stop:
                    current = checks[count]
                    generated.append(current)
                    stop = len(generated) >= c.max_new_tokens or current == eos_token_id
                if not stop:
                    committed = len(input_ids)+progress+count
                    if count == len(draft):
                        draft_attn.step(draft[-1], self.native_s)
                        sr = draft_attn.shadow
                    else:
                        sr = probes[count]
                    tr = self.observation.rows[:, count, :committed].clone()
                    sr = sr[:, :committed]
                    self.native_t.crop(committed)
                    self.native_s.crop(committed)
                self.observation.rows = None
                self.working.invalidate()
                self.injector.clear()
                trace.append(dict(round_index=len(trace), prefix=n, key=key.to_dict(), cap=cap,
                    proposed=len(draft), accepted=count, terminal=stop, workset_tokens=len(selected),
                    length_ready_at_deadline=ready, length_wait_seconds=wait, fallback=decision.fallback,
                    verify_seconds=verify_seconds if c.profile else None))
                if stop:
                    break
            self.synchronize()
            decode_end = time.perf_counter()
        except BaseException as error:
            if job is not None:
                job.abort(error)
            raise
        finally:
            try:
                self.scheduler.drain()
            finally:
                draft_attn.restore()
                self.observation.mode = None
                self.observation.rows = None
                self.working.invalidate()
                self.injector.clear()
        request_end = time.perf_counter()
        result = dict(request_id=request_id, generated=generated, tokens=len(generated),
            accepted=accepted, proposed=proposed, acceptance_rate=accepted/max(1, proposed),
            blocks=len(trace), trace=trace, prefill_seconds=prefill_end-request_start,
            decode_seconds=decode_end-prefill_end, cleanup_seconds=request_end-decode_end,
            total_seconds=request_end-request_start, decode_tps=len(generated)/(decode_end-prefill_end),
            total_tps=len(generated)/(request_end-request_start), selector_refreshes=selector.refreshes,
            working_gather_layers=self.working.gathered_layers, draft_forwards=draft_attn.steps,
            projected_positions=self.injector.projected_positions,
            schedule='async' if c.asynchronous else 'serial')
        if c.profile:
            result['cost_samples'] = dict(student_step=profile_steps, verify=profile_verify, overhead=profile_overhead)
        return result

    def close(self):
        if not self.closed:
            try:
                self.scheduler.close()
            finally:
                self.observation.restore()
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
