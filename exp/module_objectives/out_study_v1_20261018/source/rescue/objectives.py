"""Module goals: draft fidelity, useful work, reference cost and deadline wait."""
import math


def accounting(row):
    trace = row['trace']
    proposed = sum(t['proposed'] for t in trace)
    accepted = sum(t['accepted'] for t in trace)
    rejected_rounds = sum(t['accepted'] < t['proposed'] for t in trace)
    tokens = row['tokens']
    if not tokens or not trace:
        raise ValueError('A completed nonempty speculative request is required')
    if proposed != row['proposed'] or accepted != row['accepted']:
        raise ValueError('Trace totals disagree')
    waste = proposed-accepted
    return dict(committed_tokens=tokens, rounds=len(trace),
        accepted_per_round=accepted/len(trace), acceptance_rate=accepted/max(1, proposed),
        wasted_proposals=waste, avoidable_rejected_tail=waste-rejected_rounds,
        wasted_proposals_per_committed=waste/tokens,
        avoidable_tail_per_committed=(waste-rejected_rounds)/tokens,
        draft_forwards_per_committed=row['draft_forwards']/tokens,
        verify_rounds_per_100_committed=100*len(trace)/tokens,
        target_query_positions_per_committed=sum(t['proposed']+1 for t in trace)/tokens,
        mean_reference_tokens=sum(t['workset_tokens'] for t in trace)/len(trace),
        history_position_query_pairs=sum(t['proposed']*t['workset_tokens'] for t in trace),
        length_wait_seconds=sum(t.get('length_wait_seconds', 0.) for t in trace),
        deadline_ready_rounds=sum(t.get('length_ready_at_deadline') is True for t in trace),
        decode_seconds=row['decode_seconds'], total_seconds=row['total_seconds'],
        total_tps=tokens/row['total_seconds'])


def distribution_quality(logits, target):
    import torch
    lq, lp = logits.float().log_softmax(-1), target.float().log_softmax(-1)
    q, p = lq.exp(), lp.exp()
    y = int(target.argmax())
    wrong = logits.clone()
    wrong[y] = -torch.inf
    return dict(top1_matches_target=int(logits.argmax()) == y,
        teacher_argmax_probability=float(q[y]),
        teacher_argmax_nll=float(-lq[y]),
        teacher_argmax_margin=float(logits[y]-wrong.max()),
        target_kl=float((p*(lp-lq)).sum()),
        distribution_overlap=float(torch.minimum(p,q).sum()))


def horizon_outcome(frame, cap):
    """Exact one-start counterfactual, never an off-policy full-trajectory TPS."""
    if cap < 1 or cap > frame['proposed']:
        raise ValueError('Requested horizon is not observed')
    labels = frame['labels']
    accepted = 0
    eos = False
    for label in labels[:cap]:
        if label == 1:
            break
        accepted += 1
        if label == 2:
            eos = True
            break
    if accepted < cap and not eos and len(labels) <= accepted:
        raise ValueError('Unknown censored continuation')
    remaining = round(frame['features'][4]*64)
    committed = min(remaining, accepted+(not eos))
    return dict(accepted=accepted, committed=committed, waste=cap-accepted,
                avoidable_tail=max(0,cap-accepted-1), catchup=(accepted==cap and not eos and committed<remaining))


def horizon_seconds(cost, prefix, cap, outcome):
    nf = prefix/2048
    design = (1,nf,cap,cap*nf)
    return (cost.get('fixed_round',0.)+sum(a*b for a,b in zip(design,cost['verify_beta']))+
            (cap+outcome['catchup'])*cost['student_step']+cost.get('cache_update',0.))


def margin_guard_loss(logits, target, penalty, native):
    """Teacher top-1 margin, distribution fit, and preservation of native successes."""
    import torch
    import torch.nn.functional as F
    y=target.argmax()
    wrong=logits.clone()
    wrong[y]=-torch.inf
    lp,lq=target.float().log_softmax(-1),logits.float().log_softmax(-1)
    kl=(lp.exp()*(lp-lq)).sum()
    margin=F.softplus(wrong.max()-logits[y]+.1)
    native_correct=(native.argmax()==y).float()
    # Preserve a successful native margin rather than rewarding an injection
    # merely because its average KL improved on other queries.
    old_wrong=native.clone();old_wrong[y]=-torch.inf
    old_margin=(native[y]-old_wrong.max()).detach().clamp_min(0)
    preserve=native_correct*F.relu(old_margin-(logits[y]-wrong.max()))
    return .25*kl+margin+.5*preserve+.01*penalty
