"""CPU-only SYNTHETIC contracts; no actual corpus, model, GPU or efficacy."""
import copy
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import csgb_episode_runner as runner
from csgb_episode_policy import EpisodePolicy
from test_csgb_episode_harness import fixture
from test_csgb_online_harness import Clock, IDs, Model

UUID = "GPU-c7642b47-f924-b4b1-3e90-ac94a72f1830"


def protocol_state():
    return {"schema":1,"kind":"csgb_episode_train_only_pilot_v1",
        "fit_documents":list(range(16)),"monitor_documents":list(range(16,24)),"contexts":[4096],
        "development_opened":False,"confirmation_opened":False,"native_v":True,
        "no_epoch_or_seed_selection":True,"no_automatic_development_or_confirmation_launch":True,
        "method":{"seed":101,"order_seed":20261016,"stream_seed":1014096,
            "budgets":[8,16,24,32],"modes":list(runner.MODES),"temperature":1.,"learning_rate":.0005,
            "passes":2,"initial_weights":"all_zero","round_count_normalization":False,
            "gradient":"episode_return_minus_independent_baseline_times_sum_log_policy_scores"},
        "monitor":{"policies":list(runner.POLICIES),"action_streams":3,"timing_repeats":2,"warmups_per_policy":1},
        "settings":dict(runner.SETTINGS),"source_sha256":"0"*64,"scaler_sha256":"1"*64,
        "checkpoint_receipt_sha256":"2"*64,"checkpoint_training_root":"/synthetic/original_training",
        "hardware":{"physical_gpu_index":4,"expected_gpu_uuid":UUID}}


def record(mode="csgb",phase="train",document=None,*,stream=None,repeat=None,counts=None,capped=False):
    """Rebase a real fake-engine episode's prompt coordinates4→4096 only."""
    document = (0 if phase in ("train","calibration","fit_warmup","smoke_fixed","smoke_train") else 16) if document is None else document
    pass_index = 0 if phase in ("train","smoke_train") else None
    stream = (0 if phase in ("monitor","monitor_warmup") else None) if stream is None else stream
    repeat = (0 if phase == "monitor" else -1 if phase in ("monitor_warmup","fit_warmup") else None) if repeat is None else repeat
    learned = mode in runner.MODES
    seed = runner.request_seed(protocol_state(),phase,mode,document,pass_index=pass_index,stream=stream) if learned else None
    h,result,reference,seconds,p,*_ = fixture(mode,phase=phase,seed=seed or 42,counts=counts,capped=capped)
    e = h.finish(result,reference,seconds)
    identity = {"document":document,"context":4096,"phase":phase,"pass_index":pass_index,
                "stream":stream,"timing_repeat":repeat,"request_seed":seed}
    for r in e["rows"]:
        r["context"] = 4096
        for key in ("prefix","input_tokens","committed_prefix_before","committed_prefix_after"):
            r[key] += 4092
        if r["root_processed_prefix_before"]: r["root_processed_prefix_before"] += 4092
        r["provenance"] = dict(identity)
    before = p.to_state() if learned else None
    row = {"schema":1,"document":document,"context":4096,"policy":mode,"phase":phase,
        "pass_index":pass_index,"stream":stream,"timing_repeat":repeat,"request_seed":seed,
        "gpu_uuid":UUID,"uncontended_at_request_boundaries":True,"input_sha256":"0"*64,
        "eos_token_id":99,"episode":e,"parameters_before":before,"parameters_after":before,
        "baseline_return":None,"rho":None,"episode_return":None,"gradient":None,"learning_rate":None,
        "update_wall_seconds":None}
    if phase in ("train","smoke_train"):
        _,fields = runner.apply_episode_update(p,e,{"rho":2.,"baseline_returns":{str(document):1.}},document)
        row.update(fields)
    return row,{"output_token_ids":reference}


class EpisodeRunnerTests(unittest.TestCase):
    def test_all_modes_actual_core_harness_replay_and_separate_external_updates(self):
        for mode in runner.MODES:
            with self.subTest(mode=mode):
                row,ref = record(mode)
                checked = runner.validate_episode_record(row,ref,eos_token_id=99,protocol=protocol_state())
                self.assertEqual(checked["useful_n"],6)
                self.assertEqual(checked["request_tps"],6/row["episode"]["seconds"])
                self.assertEqual(row["parameters_before"]["parameter_version"],0)
                self.assertEqual(row["parameters_after"]["parameter_version"],1)
                self.assertEqual(row["episode_return"],6-2*row["episode"]["seconds"])
                for r in row["episode"]["rows"]:
                    self.assertIsNone(r["seconds"]); self.assertIsNone(r["reward"]); self.assertFalse(r["updated"])
        for mode in runner.FIXED:
            row,ref = record(mode,phase="calibration")
            runner.validate_episode_record(row,ref,eos_token_id=99)
            self.assertEqual(row["episode"]["decisions"],[])

    def test_monitor_streams_repeat_rng_reset_frozen_same_softmax_and_no_labels(self):
        for mode in runner.MODES:
            a,ref = record(mode,phase="monitor",stream=1,repeat=0)
            b,_ = record(mode,phase="monitor",stream=1,repeat=1)
            c,_ = record(mode,phase="monitor",stream=2,repeat=0)
            self.assertEqual(a["request_seed"],b["request_seed"])
            self.assertEqual(a["episode"]["decisions"],b["episode"]["decisions"])
            self.assertNotEqual(a["request_seed"],c["request_seed"])
            self.assertEqual(a["parameters_before"],a["parameters_after"])
            runner.validate_episode_record(a,ref,eos_token_id=99,protocol=protocol_state())
            changed = copy.deepcopy(a); changed["parameters_after"]["weights"][0][0] += .01
            with self.assertRaises(ValueError): runner.validate_episode_record(changed,ref,eos_token_id=99)
            changed = copy.deepcopy(a); changed["rho"] = 2.
            with self.assertRaises(ValueError): runner.validate_episode_record(changed,ref,eos_token_id=99)

    def test_real128_capped_overshoot_retained_cost_useful_n_and_eos(self):
        row,ref = record("csgb",counts=(9,)*14+(5,),capped=True)
        checked = runner.validate_episode_record(row,ref,eos_token_id=99)
        self.assertEqual(checked["useful_n"],128)
        self.assertEqual(row["episode"]["result"]["raw_total_generated"],131)
        self.assertEqual(row["episode"]["rows"][-1]["useful_committed"],2)
        row,ref = record("csgb")
        self.assertEqual(row["episode"]["result"]["output_token_ids"][-1],99)
        runner.validate_episode_record(row,ref,eos_token_id=99)
        row["episode"]["result"]["output_token_ids"][0] = 99; ref["output_token_ids"][0] = 99
        with self.assertRaises(ValueError): runner.validate_episode_record(row,ref,eos_token_id=99)

    def test_mutated_returns_gradients_params_counts_rng_anchors_or_future_state_rejected(self):
        mutations = [lambda r:r.update(episode_return=999.),lambda r:r["gradient"][0].__setitem__(0,9.),
            lambda r:r["parameters_after"]["weights"][0].__setitem__(0,9.),lambda r:r.update(request_seed=1),
            lambda r:r["episode"]["rows"][-1].update(attention_observed_timestep=0),
            lambda r:r["episode"]["rows"][-1].update(anchor_timestep=1),
            lambda r:r["episode"]["rows"][-1].update(actual_draft_nodes=4),
            lambda r:r["episode"]["rows"][-1].update(useful_committed=99),
            lambda r:r["episode"]["rows"][-1].update(updated=True),
            lambda r:r["episode"]["rows"][-1].update(terminal_commit=False),
            lambda r:r["episode"]["rows"][-1]["provenance"].update(document=24),
            lambda r:r.update(update_wall_seconds=-1.),lambda r:r.update(learning_rate=.1)]
        for i,mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                row,ref = record(); mutate(row)
                with self.assertRaises(ValueError): runner.validate_episode_record(row,ref,eos_token_id=99)
        row,ref = record("history_native")
        row["episode"]["rows"][-1]["raw_x"][1] += .1
        with self.assertRaises(ValueError): runner.validate_episode_record(row,ref,eos_token_id=99)

    def test_split_epoch_stream_and_phase_relabeling_fail_closed(self):
        for mode,phase,doc in (("csgb","train",0),("fixed8_native","calibration",0),
                              ("csgb","monitor",16),("fixed8_native","fit_warmup",0),
                              ("csgb","monitor_warmup",16),("fixed8_native","smoke_fixed",0),
                              ("csgb","smoke_train",0)):
            row,ref = record(mode,phase,doc)
            runner.validate_episode_record(row,ref,eos_token_id=99,protocol=protocol_state())
        for key,value in (("document",24),("document",True),("context",8192),("context",4096.),
                          ("phase","development"),("pass_index",2),("stream",0),("timing_repeat",0)):
            with self.subTest(key=key,value=value):
                row,ref = record(); row[key] = value
                with self.assertRaises(ValueError): runner.validate_episode_record(row,ref,eos_token_id=99)

    def test_calibration_whole_requests_rho_baselines_equal_doc_bestfixed_and_no_monitor(self):
        rows = [record(mode,"calibration",doc)[0] for doc in range(16) for mode in runner.FIXED]
        summary = runner.calibrate_fit(rows)
        # Native fake requests identical across all4 budgets: tie→smaller8.
        self.assertEqual(summary["best_fixed"],"fixed8_native")
        native8 = [r["episode"] for r in rows if r["policy"] == "fixed8_native"]
        expected_rho = sum(r["useful_n"] for r in native8)/math.fsum(r["seconds"] for r in native8)
        self.assertEqual(summary["rho"],expected_rho)
        self.assertEqual(set(summary["baseline_returns"]),{str(d) for d in range(16)})
        for r in native8:
            self.assertAlmostEqual(r["useful_n"]-summary["rho"]*r["seconds"],0.)
        for bad in (rows[:-1],rows+[rows[0]],rows+[record("fixed8_native","monitor",16)[0]]):
            with self.assertRaises(ValueError): runner.calibrate_fit(bad)

    def test_timer_includes_complete_generation_and_closing_sync_but_not_update(self):
        events,clock = [],[0.]
        def sync(): events.append("sync"); clock[0] += .25
        def generate(): events.append("prefill,features,draft,verify,terminal"); clock[0] += 2.; return {"done":True}
        result,seconds = runner.timed_request(generate,synchronize=sync,clock=lambda:clock[0])
        self.assertEqual(events,["sync","prefill,features,draft,verify,terminal","sync"])
        self.assertEqual(seconds,2.25); self.assertTrue(result["done"])
        row,_ = record(); p = EpisodePolicy.from_state(row["parameters_before"])
        before = p.to_state(); clock[0] += 100.  # Cleanup/ref/serialization do not enter returned T.
        runner.apply_episode_update(p,row["episode"],{"rho":2.,"baseline_returns":{"0":1.}},0,clock=lambda:clock[0])
        self.assertEqual(p.to_state(),before); self.assertEqual(seconds,2.25)
        with self.assertRaises(ValueError): runner.apply_episode_update(p,row["episode"],{"rho":2.,"baseline_returns":{"24":1.}},24)

    def test_order_and_streams_two_fixed_passes_no_epoch_selection(self):
        p = protocol_state()
        for pass_index in (0,1):
            order = runner.training_order(p,pass_index)
            self.assertEqual(len(order),48)
            self.assertEqual(set(order),{(m,d) for m in runner.MODES for d in range(16)})
        self.assertNotEqual(runner.training_order(p,0),runner.training_order(p,1))
        with self.assertRaises(ValueError): runner.training_order(p,2)
        for mode in runner.MODES:
            actor = runner.make_policy(mode,{"mean":[0.,0.,0.],"scale":[1.,1.,1.]})
            self.assertEqual(actor.probabilities([1.,.2,.3] if mode == "history_native" else [1.,.2,.3,.06]),(.25,)*4)
        for args in (("monitor","csgb",24,None,0),("train","csgb",0,2,None),
                     ("smoke_train","csgb",1,0,None),("monitor","csgb",True,None,0)):
            phase,mode,doc,index,stream=args
            with self.assertRaises(ValueError): runner.request_seed(p,phase,mode,doc,pass_index=index,stream=stream)

    def test_physical_gpu_boundaries_errors_warnings_contenders_and_ambiguous_identity(self):
        def response(text,err=""):
            return lambda *a,**k:SimpleNamespace(stdout=text,stderr=err)
        self.assertEqual(runner.assert_gpu_boundary(UUID,query=response(f"{UUID}, 123\n"),pid=123),UUID)
        other = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self.assertEqual(runner.assert_gpu_boundary(UUID,query=response(f"{UUID}, 123\n{other}, 99\n"),pid=123),UUID)
        for text,err in ((f"{UUID}, 123\n{UUID}, 99\n",""),("",""),(f"{other}, 123\n",""),
                         (f"{UUID},123\n{other},123\n",""),(f"{UUID},123\n","warning"),("garbage","")):
            with self.subTest(text=text,err=err),self.assertRaises(RuntimeError):
                runner.assert_gpu_boundary(UUID,query=response(text,err),pid=123)
        with self.assertRaises(subprocess.CalledProcessError):
            runner.assert_gpu_boundary(UUID,query=lambda *a,**k:(_ for _ in ()).throw(subprocess.CalledProcessError(1,"nvidia-smi")),pid=123)

    def test_protocol_sources_metadata_and_isolated24_source_are_cpu_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); code = root/"specextend"; code.mkdir()
            engine = code/"csgb_episode_runner.py"; engine.write_text("# synthetic source\n")
            helper = root/"csgb_migration_provenance.py"; helper.write_text("# synthetic helper\n")
            p = protocol_state(); p["source_manifest"] = {str(q):runner.sha(q) for q in (engine,helper)}
            path = root/"pilot.json"; path.write_text(json.dumps(p))
            with patch.object(runner,"__file__",str(engine)):
                self.assertEqual(runner.validate_pilot_protocol(path),p)
                for key,value in (("fit_documents",list(range(24))),("monitor_documents",list(range(24,32))),
                                  ("development_opened",True),("contexts",[8192])):
                    changed = copy.deepcopy(p); changed[key] = value; path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError): runner.validate_pilot_protocol(path)
                path.write_text(json.dumps(p)); (code/"extra.py").write_text("# unstaged\n")
                with self.assertRaises(ValueError): runner.validate_pilot_protocol(path)
            source = root/"train24.jsonl"
            source.write_text("\n".join(json.dumps({"text":f"synthetic{i}"}) for i in range(24)))
            self.assertEqual(len(runner.load_train24(source)),24)
            source.write_text(source.read_text()+"\n"+"THIS EXTRA RECORD MUST NOT BE PARSED")
            with self.assertRaisesRegex(ValueError,"isolated24"): runner.load_train24(source)

    def test_scaler_recomputes_all_fit_only_features_and_refuses_metadata_or_source_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); inputs = {}; observed = []
            for doc in range(16):
                raw = [1.,.3+.01*doc,.1+.01*doc,0.]; raw[-1] = raw[1]*raw[2]; observed.append(raw[1:])
                rows = [{"raw_x":None},{"raw_x":None},{"raw_x":raw,"proposal":2,
                    "attention_observed_timestep":1,"anchor_timestep":0,"terminal_commit":True}]
                row = {"document":doc,"context":4096,"policy":"fixed8_fresh","phase":"calibration",
                       "exact":True,"rounds":rows,"result":{"output_token_ids":[99]}}
                for name,value in ((f"calibration_4096_{doc}_fixed8_fresh.json",row),
                                   (f"reference_4096_{doc}.json",{"output_token_ids":[99]})):
                    path = root/name; path.write_text(json.dumps(value)); inputs[str(path)] = runner.sha(path)
            columns = list(zip(*observed)); means = [math.fsum(v)/len(v) for v in columns]
            scales = [math.sqrt(math.fsum((v-m)**2 for v in c)/len(c)) for c,m in zip(columns,means)]
            script = root/"prep.py"; script.write_text("# synthetic scale prep\n")
            s = {"schema":1,"purpose":"episode fit16 fixed population scale","context":4096,
                "fit_documents":list(range(16)),"monitor_documents_used":False,"development_opened":False,
                "confirmation_opened":False,"source_sha256":"0"*64,"count":16,"mean":means,"scale":scales,
                "input_sha256":inputs,"source_manifest":{str(script):runner.sha(script)},"original_timing_or_rho_reused":False}
            path = root/"scale.json"; path.write_text(json.dumps(s))
            self.assertEqual(runner.validate_pilot_scaler(path,"0"*64)[0],s)
            for key,value in (("fit_documents",list(range(24))),("monitor_documents_used",True),
                              ("count",17),("mean",[0.,0.,0.]),("original_timing_or_rho_reused",True)):
                changed = copy.deepcopy(s); changed[key] = value; path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): runner.validate_pilot_scaler(path,"0"*64)
            path.write_text(json.dumps(s)); script.write_text("# changed source\n")
            with self.assertRaises(ValueError): runner.validate_pilot_scaler(path,"0"*64)

    def test_cpu_main_smoke_seven_actual_fake_engine_requests_one_external_update_no_freeze(self):
        import sys
        import csgb_online_evaluate_migrated as helpers
        from csgb_episode_harness import EpisodeHarness

        class RuntimeIDs(IDs):
            def cpu(self): return self
            def numpy(self): return SimpleNamespace(tobytes=lambda:bytes(self.values))

        class RuntimeModel(Model):
            def __init__(self):
                super().__init__(Clock())
                self.tokenizer = SimpleNamespace(eos_token_id=99)
                self.base_model = SimpleNamespace(float=lambda:None)
            def eval(self): return self
            def parameters(self): return [SimpleNamespace(device="cuda:0")]
            def ar_generate(self,ids,**kwargs): return {"output_token_ids":[7,7,7,7,7,99]}
            def spgenerate(self,ids,**kwargs):
                tokens = list(ids.values); context = len(tokens)
                self.draft(RuntimeIDs(tokens+[7]),8,.7,10)
                for j in range(3):
                    tokens += [7,7]
                    if j == 2:
                        tokens[-1] = 99; self.csgb_terminal_commit_callback(RuntimeIDs(tokens))
                    else:
                        refresh = j%4 == 0
                        if refresh or self.csgb_attention_every_round:
                            self.attn_scores_final = [1.]*len(tokens)
                            self.csgb_attention_observed_timestep = j
                        else: self.attn_scores_final = None
                        self.retrieval_condition = refresh
                        self.draft(RuntimeIDs(tokens+[7]),8,.7,10)
                return {"output_token_ids":tokens[context:],"total_generated":6,"raw_total_generated":6,
                        "generation_overshoot":0,"accept_length_list":[1,1,1],"raw_accept_length_list":[1,1,1]}

        def clear(model):
            model.selected_chunks=None; model.total_seq_len=0; model.attn_scores_final=None
            model.csgb_attention_observed_timestep=None; model.retrieval_condition=False
        def harness(model):
            def feature(attention,readable,anchor):
                if anchor is None:return None
                c,d=sum(attention[i] for i in readable)/sum(attention),.25
                return [1.,c,d,c*d]
            return EpisodeHarness(model,attention_to_host=list,feature_fn=feature,
                                  synchronize=lambda:(_ for _ in ()).throw(AssertionError("round sync")))

        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,synchronize=lambda:None),
            empty=lambda *a,**k:object(),manual_seed=lambda s:None,float16="float16",device=lambda s:s,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True)),
            set_float32_matmul_precision=lambda s:None,inference_mode=nullcontext)
        modules = {"torch":torch,"numpy":SimpleNamespace(random=SimpleNamespace(seed=lambda s:None)),
            "classic.model_classic":SimpleNamespace(SPModel=SimpleNamespace(from_pretrained=lambda **k:RuntimeModel())),
            "baseline_profiler":SimpleNamespace(exact_input=lambda tokenizer,text,context,*a:RuntimeIDs([1]*context)),
            "csgb_same_state":SimpleNamespace(release_request_caches=clear)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); out=root/"smoke"
            files={name:root/name for name in ("train24.jsonl","scale.json","pilot.json","receipt.json")}
            files["train24.jsonl"].write_text("\n".join(json.dumps({"text":f"synthetic{i}"}) for i in range(24)))
            for name in ("scale.json","pilot.json","receipt.json"): files[name].write_text("{}")
            p=protocol_state();p.update(source_sha256=runner.sha(files["train24.jsonl"]),
                scaler_sha256=runner.sha(files["scale.json"]),checkpoint_receipt_sha256=runner.sha(files["receipt.json"]),
                source_manifest={str(files["pilot.json"]):runner.sha(files["pilot.json"])})
            args=["runner","--source",str(files["train24.jsonl"]),"--scaler",str(files["scale.json"]),
                "--protocol",str(files["pilot.json"]),"--checkpoint-receipt",str(files["receipt.json"]),
                "--base","syntheticbase","--draft","syntheticdraft","--out",str(out),"--seed","101",
                "--physical-gpu-index","4","--expected-gpu-uuid",UUID,"--smoke"]
            with patch.dict(sys.modules,modules),patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":UUID}),\
                 patch.object(sys,"argv",args),patch.object(runner,"validate_pilot_protocol",return_value=p),\
                 patch.object(runner,"validate_pilot_scaler",return_value=({"mean":[0.,0.,0.],"scale":[1.,1.,1.]},{})),\
                 patch.object(runner,"verify_loaded_sources"),patch.object(helpers,"validate_physical_gpu"),\
                 patch.object(helpers,"validate_checkpoint_receipt",return_value=({},{})),\
                 patch.object(runner,"wait_for_preload_gpu"),\
                 patch.object(runner,"assert_gpu_boundary"),patch.object(runner,"EpisodeHarness",side_effect=harness),\
                 patch("builtins.print"):
                runner.main()
                with self.assertRaises(FileExistsError):runner.main()
            done=runner.load(out/"DONE.json")
            self.assertTrue(done["smoke"]);self.assertEqual(done["requests_including_warmups"],7)
            self.assertFalse(done["monitor_opened"]);self.assertFalse((out/"frozen_policies.json").exists())
            raw=[runner.load(q) for q in out.glob("smoke_*.json")]
            self.assertEqual(len(raw),7)
            self.assertEqual(sum(r["phase"]=="smoke_train" for r in raw),3)
            for row in raw:
                runner.validate_episode_record(row,{"output_token_ids":[7,7,7,7,7,99]},eos_token_id=99,protocol=p)
                if row["phase"]=="smoke_train":self.assertEqual(row["parameters_after"]["parameter_version"],1)
            self.assertFalse(any(out.glob("monitor_*.json")))


if __name__ == "__main__": unittest.main()
