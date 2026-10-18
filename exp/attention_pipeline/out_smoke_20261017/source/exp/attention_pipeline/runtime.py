"""First-token overlap of length planning and selective V drafting.

Frozen selection, predictor, mapper and cost coefficients. Serial and async
controls use the same fixed `spent` input, rather than a schedule-dependent
stopwatch in the policy. This is a new execution protocol, not legacy timing.
"""
import time

import torch

from exp.attention_pipeline.cached_injection import CachedInjection
from exp.attention_pipeline.planner import PlanKey
from exp.selective_kv.bridge import Injection


@torch.no_grad()
def speculative(e, item, mode, planner, control_stream, *, planning_spent=0., delivery_delay=0.):
    # Imported lazily after the runner has installed the historical module paths.
    from experiment import sync, core_forward, tensor_ids, legacy_cache, trim, confidence
    from runtime import risk_model, features, choose_length

    if mode not in ('serial_original', 'serial_cached', 'async_cached'):
        raise ValueError('Unknown pipeline control')
    asynchronous = mode == 'async_cached'
    e.load_mapper()
    e.injection.clear()
    e.injection = Injection(e.sc) if mode == 'serial_original' else CachedInjection(e.sc)
    model = risk_model(e, 'attention_read')
    tc, sc, tr, sr, conf, prefill = e.prefill(item)
    # Runtime consumes last-layer native-key shadow, not 28 expanded rows.
    e.sc.capture = False
    generated, blocks = [], []
    current, previous = item['ids'][-1], None
    proposed = accepted = 0
    sync()
    started = time.perf_counter()
    try:
        while len(generated) < e.a.max_new_tokens:
            n = tc[-1][0].shape[-2]
            progress = len(generated)
            remaining = e.a.max_new_tokens-progress
            t, s = e.banks(tc, sc, tr, sr)
            selection = e.make_selection('read_joint1', t, s, previous)
            previous = selection
            key = PlanKey(str(item['source_index']), len(blocks), n, len(blocks))
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(e.a.device))

            # Default-argument capture makes the message's snapshot immutable
            # even if the caller later rebinds loop-local names.
            @torch.no_grad()
            def compute(t=t, s=s, selection=selection, progress=progress,
                        conf=conf, n=n, remaining=remaining, ready=ready):
                with torch.cuda.device(e.a.device), torch.cuda.stream(control_stream):
                    control_stream.wait_event(ready)
                    vector = features(e, t, s, selection, progress, 0, conf)
                    result = choose_length(e, model, vector, n, remaining, planning_spent)
                if delivery_delay:
                    time.sleep(delivery_delay)
                return result

            wait_seconds = 0.
            if asynchronous:
                ticket = planner.submit(key, remaining, compute)
                cap, decision_seconds = 1, None
            else:
                tick = time.perf_counter()
                cap, _ = compute()
                decision_seconds = time.perf_counter()-tick
                ticket = None
            # This work can overlap M3; it does not consume k.
            e.sc.configure(selection)
            if isinstance(e.injection, CachedInjection):
                e.injection.prepare(sc, selection)
            e.bind_injection(tc, selection, t)
            draft, probes = [], []
            next_input = current
            while len(draft) < cap:
                sout = e.forward(e.student, next_input, sc)
                sc = legacy_cache(sout.past_key_values)
                logits = e.logit(e.student, sout)
                conf_next = confidence(logits)
                next_input = int(logits.argmax())
                draft.append(next_input)
                probes.append((e.shadow, conf_next))
                if ticket is not None:
                    # Deadline: before a second token could exceed cap==1.
                    tick = time.perf_counter()
                    decision = ticket.result(key)
                    wait_seconds = time.perf_counter()-tick
                    cap, decision_seconds = decision.cap, decision.wall_seconds
                    ticket = None
                if next_input == e.st.eos_token_id:
                    break
            tout = core_forward(e.teacher, tensor_ids([current]+draft, e.a.device), tc, attentions=True)
            checks = e.teacher.lm_head(tout.last_hidden_state)[0, :, :e.vocab].float().argmax(-1).tolist()
            count = 0
            for j, token in enumerate(draft):
                if token != checks[j]:
                    break
                count += 1
                if token == e.st.eos_token_id:
                    break
            proposed += len(draft)
            accepted += count
            context = len(item['ids'])+progress
            generated.extend(draft[:count])
            stop = generated[-1:] == [e.st.eos_token_id] or len(generated) >= e.a.max_new_tokens
            if not stop:
                current = int(checks[count])
                generated.append(current)
                stop = current == e.st.eos_token_id or len(generated) >= e.a.max_new_tokens
            if not stop:
                committed = context+count
                if count == len(draft):
                    sout = e.forward(e.student, draft[-1], sc)
                    sc = legacy_cache(sout.past_key_values)
                    sr, conf = e.shadow, confidence(e.logit(e.student, sout))
                else:
                    sr, conf = probes[count]
                tr = tout.attentions[27][0, :, count, :committed].float().detach()
                sr = sr[:, :committed]
                tc, sc = trim(tout.past_key_values, committed), trim(sc, committed)
                assert tc[-1][0].shape[-2] == sc[-1][0].shape[-2] == tr.shape[-1] == sr.shape[-1]
            blocks.append(dict(round_index=len(blocks), prefix=n, cap=cap, proposed=len(draft),
                               accepted=count, terminal=stop, decision_wall_seconds=decision_seconds,
                               decision_wait_seconds=wait_seconds, decision_key=key.__dict__))
            if stop:
                break
        sync()
        decode = time.perf_counter()-started
    finally:
        # Drain before clearing caches/state, including an exceptional Draft.
        if planner.pending is not None:
            try:
                planner.pending.future.result()
            finally:
                e.injection.clear()
        else:
            e.injection.clear()
        e.sc.enabled = False
    return dict(source_index=item['source_index'], mode=mode, generated=generated,
                tokens=len(generated), proposed=proposed, accepted=accepted, blocks=len(blocks),
                acceptance_rate=accepted/max(1, proposed), prefill_seconds=prefill,
                decode_seconds=decode, total_seconds=prefill+decode,
                decode_tps=len(generated)/decode, total_tps=len(generated)/(prefill+decode),
                decision_wait_seconds=sum(b['decision_wait_seconds'] for b in blocks), trace=blocks)
