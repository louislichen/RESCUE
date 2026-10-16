"""New adapter against actual softmax core and a CPU fake decoding engine."""
import copy
import json
import random
import unittest
from dataclasses import FrozenInstanceError

from csgb_episode_harness import (EpisodeHarness, audit_decisions, rng_state,
                                 validate_feature_chronology, validate_request_accounting)
from csgb_episode_policy import Decision, EpisodePolicy
from test_csgb_online_harness import Clock, IDs, Model


def policy(mode):
    width = {"csgb":4,"constant":1,"history_native":3}[mode]
    return EpisodePolicy(mode,tuple((0.,)*width for _ in range(4)),1.,
                         (0.,0.,0.) if mode == "csgb" else None,
                         (1.,1.,1.) if mode == "csgb" else None)


def fixture(mode="csgb", *, phase="train", limit=128, seed=42, capped=False, counts=None):
    clock, copies = Clock(), []
    model = Model(clock)
    def sync():
        raise AssertionError("No explicit round sync during either train or deployment")
    def to_host(attention):
        copies.append(id(attention));clock.advance(.02);return list(attention)
    def features(attention,readable,anchor):
        clock.advance(.03)
        if anchor is None:return None
        c,d=sum(attention[i] for i in readable)/sum(attention),.25
        return [1.,c,d,c*d]
    h=EpisodeHarness(model,clock=clock,synchronize=sync,attention_to_host=to_host,feature_fn=features)
    actor=None if mode.startswith("fixed") else policy(mode)
    rng=None if actor is None else random.Random(seed)
    before=None if rng is None else rng_state(rng)
    h.reset(mode,actor,rng,phase=phase,context=4,max_new_tokens=limit,eos_token_id=99,
            provenance={"phase":phase,"synthetic_document":"not a real split"})
    tokens=[99,0,0,0]  # Prompt EOS must not terminate generation.
    start=clock()
    model.draft(IDs(tokens+[7]),8,.7,10)
    raw=[]
    counts=counts or ((2,2,3) if capped else (2,2,2))
    for j,count in enumerate(counts):
        tokens.extend([7]*count)
        terminal=j==len(counts)-1
        raw.append(count-1)
        clock.advance(.2)
        if terminal:
            if not capped:tokens[-1]=99
            h.finalize_terminal(IDs(tokens))
        else:
            refresh=j%4==0
            if refresh or model.csgb_attention_every_round:
                model.attn_scores_final=[1.]*(len(tokens))
                model.csgb_attention_observed_timestep=j
            else:model.attn_scores_final=None
            model.retrieval_condition=refresh
            model.draft(IDs(tokens+[7]),8,.7,10)
    output=tokens[4:4+limit]
    scored=[];generated=0
    for count in counts:
        scored.append(min(generated+count,len(output))-min(generated,len(output))-1)
        generated+=count
    result={"output_token_ids":output,"accept_length_list":scored,"raw_accept_length_list":raw,
            "total_generated":len(output),"raw_total_generated":sum(a+1 for a in raw),
            "generation_overshoot":sum(a+1 for a in raw)-len(output)}
    return h,result,list(output),clock()-start,actor,rng,before,copies,model


class EpisodeHarnessTests(unittest.TestCase):
    def test_real_core_one_decision_per_verified_round_fallback_consumes_no_rng_and_json_replays(self):
        h,result,reference,seconds,p,rng,before,copies,model=fixture()
        state_before=p.to_state()
        summary=h.finish(result,reference,seconds)
        self.assertEqual(summary["phase"],"train")
        self.assertEqual(summary["useful_n"],6)
        self.assertEqual(summary["request_tps"],6/seconds)
        self.assertEqual(len(summary["decisions"]),3)
        decisions=[Decision.from_state(s) for s in summary["decisions"]]
        self.assertEqual([d.raw_x is None for d in decisions],[True,True,False])
        self.assertEqual([d.u for d in decisions[:2]],[None,None])
        expected=random.Random(42);expected.random()
        self.assertEqual(rng_state(expected),rng_state(rng))
        self.assertEqual(summary["rng_before"],before)
        self.assertNotEqual(summary["rng_before"],summary["rng_after"])
        self.assertEqual(p.to_state(),state_before)
        saved=json.loads(json.dumps(summary))
        self.assertEqual(audit_decisions(p,saved["decisions"],saved["rows"],saved["rng_before"],saved["rng_after"]),saved["decisions"])
        self.assertEqual(len(copies),2)
        self.assertEqual(len(model.drafted),3)
        self.assertIsNone(h.pending)
        self.assertTrue(h.finished)
        self.assertFalse(summary["round_updates"])
        self.assertFalse(summary["baseline_or_sgd_performed"])
        for row in summary["rows"]:
            self.assertIsNone(row["seconds"]);self.assertIsNone(row["reward"]);self.assertFalse(row["updated"])

    def test_train_and_deployment_share_identical_sampling_no_greedy_or_phase_relabel(self):
        a=fixture(phase="train");b=fixture(phase="development")
        sa=a[0].finish(a[1],a[2],a[3]);sb=b[0].finish(b[1],b[2],b[3])
        self.assertEqual(sa["decisions"],sb["decisions"])
        self.assertEqual(sa["rng_after"],sb["rng_after"])
        self.assertEqual((sa["phase"],sb["phase"]),("train","development"))
        self.assertEqual(sa["parameter_sha256"],sb["parameter_sha256"])

    def test_constant_and_native_history_keep_original_observation_paths(self):
        for mode in ("constant","history_native"):
            with self.subTest(mode=mode):
                h,result,reference,seconds,p,rng,before,copies,model=fixture(mode)
                summary=h.finish(result,reference,seconds)
                self.assertEqual(len(copies),2 if mode=="constant" else 0)
                self.assertEqual(model.csgb_attention_every_round,mode=="constant")
                final=Decision.from_state(summary["decisions"][-1])
                self.assertEqual(len(final.transformed_x),1 if mode=="constant" else 3)
                self.assertEqual(h.learner.__class__.__name__,"_Bridge")

    def test_native_fixed_uses_same_clean_path_without_actor_or_rng(self):
        for budget in (8,16,24,32):
            with self.subTest(budget=budget):
                h,result,reference,seconds,p,rng,before,copies,model=fixture(f"fixed{budget}_native")
                summary=h.finish(result,reference,seconds)
                self.assertEqual(model.drafted,[budget]*3)
                self.assertEqual(copies,[])
                self.assertEqual(summary["decisions"],[])
                self.assertIsNone(summary["parameter_sha256"])
                self.assertIsNone(summary["rng_before"])

    def test_feature_matched_fixed8_calibration_is_not_mislabeled_as_training_or_development(self):
        h,result,reference,seconds,p,rng,before,copies,model=fixture("fixed8_fresh",phase="calibration")
        summary=h.finish(result,reference,seconds)
        self.assertEqual(summary["phase"],"calibration")
        self.assertEqual(summary["decisions"],[])
        self.assertEqual(len(copies),2)
        self.assertTrue(model.csgb_attention_every_round)

    def test_final_output_limit_overshoot_is_charged_but_n_and_acceptance_are_clipped(self):
        h,result,reference,seconds,*_=fixture(limit=5,capped=True)
        summary=h.finish(result,reference,seconds)
        self.assertEqual(summary["useful_n"],5)
        self.assertEqual(summary["scored_acceptance"],2/3)
        self.assertEqual(summary["rows"][-1]["raw_committed"],3)
        self.assertEqual(summary["rows"][-1]["useful_committed"],1)
        self.assertTrue(summary["rows"][-1]["terminal_commit"])

    def test_full128_limit_one_decision_per_verified_round_and_external_next_request_reset(self):
        h,result,reference,seconds,p,rng,*_=fixture(capped=True,counts=(9,)*14+(5,))
        summary=h.finish(result,reference,seconds)
        self.assertEqual((summary["useful_n"],result["raw_total_generated"]),(128,131))
        self.assertEqual(len(summary["decisions"]),15)
        self.assertEqual(summary["rows"][-1]["useful_committed"],2)
        self.assertEqual(summary["result"],result)
        self.assertEqual(summary["reference_output_token_ids"],reference)
        h.reset("csgb",p,random.Random(43),phase="train",context=4)
        self.assertEqual(h.rows,[])
        self.assertEqual(h.bridge.decisions,[])
        self.assertFalse(h._episode_finalized)

    def test_rng_extra_draw_decision_raw_transform_budget_and_parameters_mutations_rejected(self):
        for kind in ("rng","decision","raw","transform","budget","index","parameters"):
            with self.subTest(kind=kind):
                h,result,reference,seconds,p,rng,*_=fixture()
                if kind=="rng":rng.random()
                elif kind=="decision":h.bridge.decisions.pop()
                elif kind=="raw":h.rows[-1]["raw_x"]=[1.,.5,.25,.125]
                elif kind=="transform":h.rows[-1]["learner_input"][1]+=.1
                elif kind=="budget":h.rows[-1]["budget"]=8
                elif kind=="index":h.rows[-1]["episode_decision_index"]=True
                else:object.__setattr__(p,"temperature",2.)
                with self.assertRaises(ValueError):h.finish(result,reference,seconds)
                self.assertFalse(h._episode_finalized)

    def test_token_round_no_update_terminal_and_canonical_fail_closed(self):
        for kind in ("canonical","EOS","early","roundcount","actualnodes","root","update","roundseconds","terminal","seconds","short_seconds"):
            with self.subTest(kind=kind):
                h,result,reference,seconds,*_=fixture()
                if kind=="canonical":reference[0]=8
                elif kind=="EOS":result["output_token_ids"][0]=99;reference[0]=99
                elif kind=="early":result["output_token_ids"][-1]=7;reference[-1]=7
                elif kind=="roundcount":h.rows.pop()
                elif kind=="actualnodes":h.rows[0]["actual_draft_nodes"]+=1
                elif kind=="root":h.rows[0]["useful_committed"]-=1
                elif kind=="update":h.rows[0]["updated"]=True
                elif kind=="roundseconds":h.rows[0]["seconds"]=.1
                elif kind=="terminal":h.rows[0]["terminal_commit"]=True
                elif kind=="seconds":seconds=float("nan")
                else:seconds=1e-12
                with self.assertRaises(ValueError):h.finish(result,reference,seconds)

    def test_unfinished_extra_unverified_final_and_duplicate_finish_cannot_be_discarded(self):
        h,result,reference,seconds,p,rng,*_=fixture()
        h.finished=False
        with self.assertRaises(RuntimeError):h.finish(result,reference,seconds)
        with self.assertRaises(RuntimeError):h.reset("csgb",p,rng,phase="train",context=4)
        h.finished=True;h.pending={"unverified":True}
        with self.assertRaises(RuntimeError):h.finish(result,reference,seconds)
        h.pending=None;h.bridge.decisions.append(p.sample(None,rng))
        with self.assertRaises(ValueError):h.finish(result,reference,seconds)
        h.bridge.decisions.pop();h.finish(result,reference,seconds)
        with self.assertRaises(RuntimeError):h.finish(result,reference,seconds)

    def test_reset_mode_rng_phase_guard_and_immutable_parameters(self):
        clock=Clock();h=EpisodeHarness(Model(clock),clock=clock)
        p=policy("csgb");rng=random.Random(3)
        with self.assertRaises(FrozenInstanceError):p.temperature=2.
        for options in ({"phase":"development","provenance":{"phase":"train"}},
                        {"phase":"train","provenance":[]},{"phase":""}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                h.reset("csgb",p,rng,context=4,**options)
        with self.assertRaises(ValueError):h.reset("constant",p,rng,phase="train",context=4)
        with self.assertRaises(ValueError):h.reset("csgb",p,None,phase="train",context=4)
        with self.assertRaises(ValueError):h.reset("fixed8_native",p,rng,phase="train",context=4)

    def test_feature_chronology_all_paths_and_real_refresh4_root_suffix(self):
        for mode in ("csgb","constant","history_native","fixed8_fresh",
                     "fixed8_native","fixed16_native","fixed24_native","fixed32_native"):
            with self.subTest(mode=mode):
                h,result,reference,seconds,*_=fixture(mode,counts=(2,)*11)
                checked=validate_feature_chronology(h.rows,mode)
                self.assertEqual(checked["verified_rounds"],11)
                self.assertEqual(checked["fresh_feature_rounds"],9 if mode in ("csgb","constant","fixed8_fresh") else 0)
                self.assertFalse(checked["full_attention_reconstructed"])
                self.assertEqual([r["anchor_timestep"] for r in h.rows],
                                 [None,None,0,0,0,0,4,4,4,4,8])
                self.assertEqual([r["root_processed_prefix_before"] for r in h.rows],
                                 [0]+[r["prefix"] for r in h.rows[:-1]])
                h.finish(result,reference,seconds)

    def test_feature_chronology_rejects_readiness_anchor_timestamp_and_root_tampering(self):
        mutations=(lambda r:r[0].update(root_processed_prefix_before=1),
                   lambda r:r[2].update(root_processed_prefix_before=r[2]["prefix"]-1),
                   lambda r:r[2].update(prefix=r[2]["prefix"]+1),
                   lambda r:r[2].update(input_tokens=r[2]["input_tokens"]+1),
                   lambda r:r[2].update(context=8),
                   lambda r:r[2].update(has_real_anchor=False),
                   lambda r:r[2].update(has_real_anchor=1),
                   lambda r:r[2].update(eligible_for_calibration=False),
                   lambda r:r[2].update(anchor_timestep=1),
                   lambda r:r[2].update(anchor_timestep=False),
                   lambda r:r[1].update(anchor_timestep=0),
                   lambda r:r[0].update(attention_observed_timestep=0),
                   lambda r:r[2].update(attention_observed_timestep=0),
                   lambda r:r[2].update(attention_observed_timestep=True),
                   lambda r:r[2].pop("coverage"))
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                h,result,reference,seconds,*_=fixture()
                mutate(h.rows)
                with self.assertRaises(ValueError):validate_feature_chronology(h.rows,"csgb")
                with self.assertRaises(ValueError):h.finish(result,reference,seconds)

    def test_feature_chronology_rejects_fabricated_raw_product_or_startup_features(self):
        for kind in ("x","product","coverage","shift","boolean","readiness_missing",
                     "startup_raw","startup_coverage","startup_input","transformed"):
            with self.subTest(kind=kind):
                h,result,reference,seconds,*_=fixture()
                row=h.rows[-1]
                if kind=="x":row["x"]=[1.,.5,.25,.125]
                elif kind in ("product","boolean"):
                    value=list(row["raw_x"])
                    value[3] += .01
                    if kind=="boolean":value=[1.,True,.25,.25]
                    for key in ("raw_x","x","raw_features"):row[key]=list(value)
                elif kind=="coverage":row["coverage"]+=.01
                elif kind=="shift":row["shift"]+=.01
                elif kind=="readiness_missing":
                    for key in ("raw_x","x","raw_features","coverage","shift","learner_input"):row[key]=None
                elif kind=="startup_raw":
                    for key in ("raw_x","x","raw_features"):h.rows[0][key]=list(row["raw_x"])
                elif kind=="startup_coverage":h.rows[0]["coverage"]=0.
                elif kind=="startup_input":h.rows[0]["learner_input"]=[1.,0.,0.,0.]
                else:row["learner_input"]=None
                with self.assertRaises(ValueError):validate_feature_chronology(h.rows,"csgb")

    def test_native_history_is_prior_useful_acceptance_not_raw_or_future(self):
        h,result,reference,seconds,*_=fixture("history_native",counts=(1,3,2))
        self.assertEqual(h.rows[-1]["raw_x"],[1.,.2,.05])
        validate_feature_chronology(h.rows,"history_native")
        for kind in ("future","ema","fresh","timestamp","coverage","input"):
            rows=copy.deepcopy(h.rows);row=rows[-1]
            if kind in ("future","ema"):
                value=list(row["raw_x"]);value[1 if kind=="future" else 2]+=.01
                row["raw_x"]=row["x"]=row["learner_input"]=value
            elif kind=="fresh":row["raw_features"]=[1.,.5,.25,.125]
            elif kind=="timestamp":row["attention_observed_timestep"]=1
            elif kind=="coverage":row["coverage"]=.5
            else:row["learner_input"]=[1.,.2,.1]
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_feature_chronology(rows,"history_native")
        h.finish(result,reference,seconds)

    def test_native_fixed_has_no_features_but_fresh_calibration_has_raw_input(self):
        h,*_=fixture("fixed8_native")
        for key in ("raw_x","x","raw_features","coverage","shift","attention_observed_timestep","learner_input"):
            rows=copy.deepcopy(h.rows)
            rows[-1][key]=[1.,.5,.25,.125] if key in ("raw_x","x","raw_features","learner_input") else 0
            with self.subTest(key=key),self.assertRaises(ValueError):validate_feature_chronology(rows,"fixed8_native")
        fresh,*_=fixture("fixed8_fresh")
        self.assertEqual(fresh.rows[-1]["learner_input"],fresh.rows[-1]["raw_x"])
        fresh.rows[-1]["learner_input"]=[1.,0.,0.,0.]
        with self.assertRaises(ValueError):validate_feature_chronology(fresh.rows,"fixed8_fresh")
        for rows,mode in (([],"csgb"),([{}],"csgb"),(h.rows,"unknown")):
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate_feature_chronology(rows,mode)


if __name__ == "__main__":
    unittest.main()
