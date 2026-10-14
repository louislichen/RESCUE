"""Critical terminal/censoring checks independent of any model forward."""
import itertools
from types import SimpleNamespace

import torch

from risk import outcome_labels
from runtime import choose_length


def check():
    eos=99
    labels,mask=outcome_labels([11,22,eos],3,eos)
    assert labels[:3]==[0,0,1] and sum(mask)==3
    labels,mask=outcome_labels([11,22,33],1,eos)
    assert labels[:2]==[0,2] and sum(mask)==2
    labels,mask=outcome_labels([11,22],2,eos)
    assert labels[:2]==[0,0] and sum(mask)==2
    # Independently enumerate a competing-event tree. Both accepted EOS and
    # rejection contribute one terminal token, but have different labels.
    hazards=[(.8,.1,.1),(.75,.1,.15),(1/3,1/3,1/3)]
    survival=torch.tensor([.8,.6,.2]+[.1]*9)
    class ToyRisk:
        def continuation(self,x):return survival
    expected={}
    for k in (1,2,3):
        total=0.
        for outcomes in itertools.product(range(3),repeat=k):
            probability=1.;committed=0
            for j,outcome in enumerate(outcomes):
                probability*=hazards[j][outcome]
            for outcome in outcomes:
                committed+=1
                if outcome:break
            else:
                committed+=1
            total+=probability*min(3,committed)
        expected[k]=total
    assert abs(expected[3]-2.4)<1e-6
    e=SimpleNamespace(cost=dict(verify_beta=[6.,0.,0.,0.],student_step=1.,cache_update=0.))
    cap,score=choose_length(e,ToyRisk(),torch.zeros(31),1000,3,0.)
    costs={k:6+k+(float(survival[k-1]) if k<3 else 0) for k in expected}
    oracle=max(expected,key=lambda k:expected[k]/costs[k])
    assert cap==oracle==2
    assert abs(score-expected[cap]/costs[cap])<1e-6
    return dict(eos_separate_from_rejection=True,unobserved_tail_censored=True,
                output_budget_expectation_matches_event_tree=True,bonus_catchup_omitted_at_budget=True)


if __name__=='__main__':
    print(check())
