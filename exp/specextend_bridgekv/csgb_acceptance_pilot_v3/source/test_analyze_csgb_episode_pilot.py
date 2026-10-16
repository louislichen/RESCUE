"""Synthetic CPU-only analyzer contracts; no real corpus/model/GPU/results."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/"specextend"),str(ROOT/"specextend"/"tests")]
import analyze_csgb_episode_pilot as a
import csgb_episode_runner as runner
from csgb_episode_policy import EpisodePolicy
from test_csgb_episode_runner import record, protocol_state, UUID


def rows_for_monitor():
    rows=[]
    for d in a.MONITOR:
        for p in a.POLICIES:
            for s in a.STREAMS:
                for r in a.REPEATS:
                    rows.append({"document":d,"context":4096,"policy":p,"phase":"monitor",
                        "stream":s,"timing_repeat":r,"request_seed":100+s if p in a.MODES else None,
                        "input_sha256":str(d),"trajectory_sha256":str(s),"tps":10.+10*s+4*r,
                        "acceptance":float(s+1),"decision_ms":.1})
    return rows


class SyntheticRoot:
    """Real core/harness request replay; only heavyweight source/scaler preflight mocked."""
    def __init__(self,root,smoke=True):
        self.base=Path(root).resolve();self.root=self.base/"raw";self.root.mkdir()
        self.scale={"mean":[0.,0.,0.],"scale":[1.,1.,1.]}
        def write(path,data):
            path.write_text(json.dumps(data));return str(path),a.sha(path)
        self.write=lambda name,data:write(self.root/name,data)
        method=protocol_state();inputs={}
        from csgb_online_evaluate_migrated import CHECKPOINT_NAMES, STAGE_SHA256, VERSIONS
        weights={}
        for folder,names in CHECKPOINT_NAMES.items():
            (self.base/folder).mkdir()
            for name in names:
                path=self.base/folder/name;path.write_text("synthetic "+folder+name);weights[str(path)]=a.sha(path)
        receipt={"schema":1,"training_root":method["checkpoint_training_root"],"training_stage_sha256":STAGE_SHA256,
            "versions":VERSIONS,"no_model_files_modified":True,"scope":"NOT a retroactive cryptographic prelaunch proof",
            "file_sha256":weights}
        for name,value,key in (("train24.jsonl",{"synthetic":True},"source_sha256"),
                ("scale.json",self.scale,"scaler_sha256"),("receipt.json",receipt,"checkpoint_receipt_sha256")):
            path,h=write(self.base/name,value);inputs[path]=h;method[key]=h
        engine=self.base/"synthetic.py";engine.write_text("# SYNTHETIC no model\n")
        method["source_manifest"]={str(engine):a.sha(engine)}
        for module in (runner,sys.modules["csgb_episode_policy"],sys.modules["csgb_episode_harness"]):
            path=str(Path(module.__file__).resolve());method["source_manifest"][path]=a.sha(path)
        path,h=write(self.base/"pilot_protocol.json",method);inputs[path]=h
        self.method=method
        self.write("protocol.json",{"runtime_version":1,"phase":"episode_smoke" if smoke else "episode_train_only_pilot",
            "smoke":smoke,"pilot_protocol":method,"input_manifest":inputs,"source_manifest":method["source_manifest"],
            "checkpoint_manifest":weights,"scaler_evidence":{str(engine):a.sha(engine)},
            "gpu_uuid":UUID,"physical_gpu_index":4,"production_speed_claim":False,
            "development_opened":False,"confirmation_opened":False})
        if smoke:self.fill_smoke()

    def fill_smoke(self):
        fixed=[]
        for mode in a.FIXED:
            row,ref=record(mode,"smoke_fixed");fixed.append(row)
            self.write("reference_4096_0.json",ref);self.write(a.filename(("smoke_fixed",0,mode,None,None,None)),row)
        rho=fixed[0]["episode"]["useful_n"]/fixed[0]["episode"]["seconds"]
        for mode in a.MODES:
            row,ref=record(mode,"smoke_train");p=EpisodePolicy.from_state(row["parameters_before"])
            _,fields=runner.apply_episode_update(p,row["episode"],{"rho":rho,"baseline_returns":{"0":0.}},0)
            row.update(fields);self.write(a.filename(("smoke_train",0,mode,0,None,None)),row)
        self.finish(smoke=True)

    def finish(self,smoke):
        data={"raw_data_complete":True,"smoke":smoke,"requests_including_warmups":7 if smoke else 556,
            "raw_sha256":{str(p):a.sha(p) for p in self.root.glob("*.json") if p.name!="DONE.json"},
            "no_final_models":True,"monitor_opened":False,"development_opened":False,"confirmation_opened":False,
            "study_complete":False,"production_speed_claim":False}
        if not smoke:data.update(calibration_requests=64,training_requests=96,monitor_requests=336,
                                warmup_requests=60,frozen_policies_sha256=a.sha(self.root/"frozen_policies.json"))
        self.write("DONE.json",data)

    def fill_pilot(self):
        def save(mode,phase,doc,*,policy=None,epoch=None,stream=None,repeat=None,calibration=None):
            patches=patch("test_csgb_episode_harness.policy",return_value=policy) if policy else ExitStack()
            with patches:row,ref=record(mode,phase,doc,stream=stream,repeat=repeat)
            if epoch is not None:
                # Regenerate with the pass-specific request RNG, not relabelled decisions.
                seed=runner.request_seed(self.method,phase,mode,doc,pass_index=epoch,stream=stream)
                from test_csgb_episode_harness import fixture
                with patch("test_csgb_episode_harness.policy",return_value=policy):
                    h,result,reference,seconds,*_=fixture(mode,phase=phase,seed=seed)
                    e=h.finish(result,reference,seconds)
                for r in e["rows"]:
                    r["context"]=4096
                    for k in ("prefix","input_tokens","committed_prefix_before","committed_prefix_after"):r[k]+=4092
                    if r["root_processed_prefix_before"]:r["root_processed_prefix_before"]+=4092
                row.update(episode=e,request_seed=seed,pass_index=epoch)
                identity={k:row[k] for k in ("document","context","phase","pass_index","stream","timing_repeat","request_seed")}
                for r in e["rows"]:r["provenance"]=dict(identity)
            updated=policy
            if calibration is not None:
                updated,fields=runner.apply_episode_update(policy,row["episode"],calibration,doc);row.update(fields)
            self.write(f"reference_4096_{doc}.json",ref)
            key=phase,doc,mode,row["pass_index"],row["stream"],row["timing_repeat"]
            self.write(a.filename(key),row)
            return row,updated
        for mode in a.FIXED:save(mode,"fit_warmup",0)
        calrows=[save(mode,"calibration",doc)[0] for doc in a.FIT for mode in a.FIXED]
        cal=runner.calibrate_fit(calrows)
        cal["request_sha256"]={str(p):a.sha(p) for p in self.root.glob("calibration_*.json")}
        self.write("calibration.json",cal)
        self.write("training_order.json",{"passes":[runner.training_order(self.method,p) for p in (0,1)]})
        current={m:runner.make_policy(m,self.scale) for m in a.MODES}
        for epoch in (0,1):
            for mode,doc in runner.training_order(self.method,epoch):
                _,current[mode]=save(mode,"train",doc,policy=current[mode],epoch=epoch,calibration=cal)
        protocol=a.load(self.root/"protocol.json")
        self.write("frozen_policies.json",{"models":{m:p.to_state() for m,p in current.items()},
            "final_only_no_selection":True,"passes":2,"fit_documents":list(a.FIT),
            "monitor_documents_used_for_training":False,"calibration_sha256":a.sha(self.root/"calibration.json"),
            "scaler_sha256":self.method["scaler_sha256"],"source_manifest":protocol["source_manifest"],
            "input_manifest":protocol["input_manifest"],"development_opened":False,"confirmation_opened":False,
            "training_evidence":{str(p):a.sha(p) for p in self.root.glob("*.json")}})
        for doc in a.MONITOR:
            for mode in a.POLICIES:save(mode,"monitor_warmup",doc,policy=current.get(mode))
            for mode in a.POLICIES:
                for stream in a.STREAMS:
                    for repeat in a.REPEATS:save(mode,"monitor",doc,policy=current.get(mode),stream=stream,repeat=repeat)
        self.finish(smoke=False)

    def patches(self):
        stack=ExitStack()
        stack.enter_context(patch.object(runner,"validate_pilot_protocol",return_value=self.method))
        stack.enter_context(patch.object(runner,"verify_loaded_sources"))
        stack.enter_context(patch.object(runner,"validate_pilot_scaler",return_value=(self.scale,{str(self.base/"synthetic.py"):a.sha(self.base/"synthetic.py")})))
        return stack


class AnalyzerTests(unittest.TestCase):
    def test_independent_episode_rng_advances_replays_and_external_sgd(self):
        for mode in a.MODES:
            row,ref=record(mode)
            result=a.audit_episode_record(row,ref,eos_token_id=99)
            self.assertEqual(result["output_tokens"],6)
            self.assertNotEqual(row["episode"]["rng_before"],row["episode"]["rng_after"])
            self.assertNotEqual(row["parameters_before"],row["parameters_after"])
        for mutation in (lambda r:r.update(gradient=[[0.]]*4),
                lambda r:r["episode"]["rows"][-1].update(anchor_timestep=7),
                lambda r:r["episode"]["rng_after"]["state"].__setitem__(1,0),
                lambda r:r["episode"]["result"]["output_token_ids"].__setitem__(0,98)):
            row,ref=record();mutation(row)
            with self.assertRaises(ValueError):a.audit_episode_record(row,ref,eos_token_id=99)
        row,ref=record("csgb","monitor");row["rho"]=2.
        with self.assertRaises(ValueError):a.audit_episode_record(row,ref,eos_token_id=99)

    def test_registered_matrix_counts_and_no_extra_split(self):
        self.assertEqual(len(a.identities()),556);self.assertEqual(len(a.identities(True)),7)
        self.assertEqual({k[1] for k in a.identities() if k[0]=="monitor"},set(range(16,24)))
        self.assertFalse(any(k[1]>=24 for k in a.identities()))

    def test_document_unit_aggregation_streams_and_timing_repeats(self):
        s=a.aggregate_monitor(rows_for_monitor(),16)
        self.assertTrue(s["monitor_matrix_complete"]);self.assertEqual(s["primary_baseline"],"fixed16_native")
        self.assertEqual(s["absolute_metrics"]["csgb"]["tps"]["mean"],22.)
        self.assertEqual(s["comparisons_all_fixed"]["fixed16_native"]["csgb"]["tps"]["documents"],8)
        self.assertEqual(s["comparisons_all_fixed"]["fixed16_native"]["csgb"]["tps"]["ci95"],[0.,0.])
        self.assertEqual(s["action_stream_variability"]["csgb"]["16"]["tps"]["streams"],[12.,22.,32.])
        self.assertFalse(s["goal_achieved"])
        rows=rows_for_monitor()[:-1]
        partial=a.aggregate_monitor(rows,8)
        self.assertEqual(partial["complete_document_units"],list(range(16,23)))
        self.assertEqual(partial["incomplete_document_units"],[23]);self.assertEqual(len(partial["missing_cells"]),1)

    def test_duplicates_seed_reuse_and_repeat_mismatch_rejected(self):
        rows=rows_for_monitor()
        for bad in (rows+[rows[0]],):
            with self.assertRaises(ValueError):a.aggregate_monitor(bad,8)
        for key,value in (("trajectory_sha256","other"),("request_seed",999)):
            rows=rows_for_monitor();rows[-1][key]=value
            with self.assertRaises(ValueError):a.aggregate_monitor(rows,8)
        rows=rows_for_monitor()
        for r in rows:
            if r["policy"]=="csgb":r["request_seed"]=42
        with self.assertRaises(ValueError):a.aggregate_monitor(rows,8)
        with self.assertRaises(ValueError):a.aggregate_monitor([],"best_monitor")

    def test_fit_only_rho_baseline_bestfixed_tie_and_reject_monitor(self):
        rows=[]
        for d in a.FIT:
            for p in a.FIXED:
                rows.append({"document":d,"policy":p,"phase":"calibration","output_tokens":128,
                    "seconds":2. if p=="fixed8_native" else 1.,"tps":64. if p=="fixed8_native" else 128.})
        s=a.fit_calibration(rows)
        self.assertEqual(s["best_fixed"],"fixed16_native");self.assertEqual(s["rho"],64.)
        self.assertEqual(set(s["baseline_returns"].values()),{0.})
        for bad in (rows[:-1],rows+[rows[0]],[dict(r,phase="monitor") for r in rows]):
            with self.assertRaises(ValueError):a.fit_calibration(bad)

    def test_actual_smoke_full_root_cpu_replay_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SyntheticRoot(tmp)
            with f.patches():result=a.audit_root(f.root,smoke=True)
            self.assertTrue(result["summary"]["smoke_passed"]);self.assertFalse(result["summary"]["goal_achieved"])
            self.assertIsNone(result["summary"]["confidence_intervals"])
            self.assertEqual(result["summary"]["requests"],7)
            # Render is mocked here; a separate real CPU-render test below.
            def fake_render(audit,out):
                (out/"acceptance.png").write_bytes(b"SYNTHETIC");return ["acceptance"]
            with patch.object(a,"render_figures",side_effect=fake_render):manifest=a.write_analysis(result,Path(tmp)/"analysis")
            self.assertTrue(a.load(Path(tmp)/"analysis"/"summary.json")["smoke_passed"])
            self.assertEqual(manifest["artifacts"]["summary.json"],a.sha(Path(tmp)/"analysis"/"summary.json"))
            with self.assertRaises(ValueError):a.write_analysis(result,Path(tmp)/"analysis")

    def test_fresh_process_loads_provenance_helpers_before_loaded_source_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SyntheticRoot(tmp)
            code="""import sys,json
from pathlib import Path
from unittest.mock import patch
import analyze_csgb_episode_pilot as a
assert 'csgb_online_evaluate_migrated' not in sys.modules
root=Path(sys.argv[1]);runtime=a.load(root/'protocol.json');method=runtime['pilot_protocol']
scale=next(a.load(p) for p,h in runtime['input_manifest'].items() if h==method['scaler_sha256'])
def imported(mapping):
    assert 'csgb_online_evaluate_migrated' in sys.modules
    assert 'csgb_migration_provenance' in sys.modules
with patch.object(a.runner,'validate_pilot_protocol',return_value=method),patch.object(a.runner,'validate_pilot_scaler',return_value=(scale,runtime['scaler_evidence'])),patch.object(a.runner,'verify_loaded_sources',side_effect=imported):
    assert a.audit_root(root,smoke=True)['summary']['smoke_passed'] is True
"""
            subprocess.run([sys.executable,"-c",code,str(f.root)],cwd=ROOT,check=True,capture_output=True,text=True)

    def test_smoke_cli_phase_scope_hash_missing_and_canonical_fail_closed(self):
        for mutation in ("phase","smoke","source","missing","canonical"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                f=SyntheticRoot(tmp)
                if mutation in {"phase","smoke"}:
                    p=a.load(f.root/"protocol.json");p[mutation]="episode_train_only_pilot" if mutation=="phase" else False;f.write("protocol.json",p)
                elif mutation=="source":(Path(next(iter(f.method["source_manifest"])))).write_text("changed")
                elif mutation=="missing":(f.root/a.filename(next(iter(a.identities(True))))).unlink()
                else:
                    path=next(f.root.glob("smoke_train*"));r=a.load(path);r["episode"]["result"]["output_token_ids"][0]=98;f.write(path.name,r)
                with f.patches(),self.assertRaises(ValueError):a.audit_root(f.root,smoke=True)

    def test_full556_actual_core_harness_external_updates_and_monitor_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SyntheticRoot(tmp,smoke=False);f.fill_pilot()
            with f.patches():result=a.audit_root(f.root)
            s=result["summary"]
            self.assertTrue(s["analysis_complete"]);self.assertEqual(len(result["records"]),556)
            self.assertEqual(len(result["training"]),96);self.assertEqual(s["complete_document_units"],list(a.MONITOR))
            self.assertEqual(s["primary_baseline"],"fixed8_native")
            self.assertTrue(s["continuation_gate"]["evaluated"]);self.assertFalse(s["continuation_gate"]["passed"])
            self.assertFalse(s["goal_achieved"])

    def test_render_synthetic_smoke_and_monitor_and_standalone_source(self):
        try:import matplotlib
        except ImportError:self.skipTest("Matplotlib unavailable; use isolated synthetic QA plotting environment")
        base=Path(tempfile.mkdtemp(prefix="csgb_episode_synthetic_qa_"))
        f=SyntheticRoot(base)
        with f.patches():smoke=a.audit_root(f.root,smoke=True)
        smoke["synthetic_qa"]=True
        manifest=a.write_analysis(smoke,base/"smoke_figures")
        for name in manifest["figures"]:
            self.assertGreater((base/"smoke_figures"/(name+".png")).stat().st_size,1000)
        subprocess.run([sys.executable,str(base/"smoke_figures"/"figure_source.py"),"--out",str(base/"standalone_rerender")],check=True,capture_output=True)
        synthetic_rows=rows_for_monitor()
        for row in synthetic_rows:
            # Plot-only synthetic nonzero paired effects; never audit as real raw.
            effect={"csgb":(row["document"]-19)/50.,"constant":-.03,"history_native":.02}.get(row["policy"],0.)
            row["tps"]*=1+effect;row["acceptance"]+=effect*4
        sampled=[{"raw_x":[1.,.8+.004*i,.01+.001*i,(.8+.004*i)*(.01+.001*i)],
                  "budget":a.BUDGETS[i%4]} for i in range(40)]
        pilot={"summary":a.aggregate_monitor(synthetic_rows,8),
               "records":[{"policy":"csgb","phase":"monitor","timing_repeat":0,"rounds":sampled}],"training":[],
               "root":"SYNTHETIC_NO_REAL_DATA","input_sha256":{},"synthetic_qa":True}
        pilot["summary"].update(smoke=False,analysis_complete=True,continuation_gate={"evaluated":True,"passed":False})
        pilot["training"]=[{"policy":m,"episode_return":float(i),"advantage":float(i)-1} for m in a.MODES for i in range(32)]
        a.write_analysis(pilot,base/"monitor_figures")
        self.assertEqual(len(list((base/"monitor_figures").glob("*.png"))),6)
        print("SYNTHETIC CPU QA figures retained at "+str(base),flush=True)

    def test_partial_calibration_has_no_monitor_gate_or_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SyntheticRoot(tmp,smoke=False);row,ref=record("fixed8_native","calibration")
            f.write("reference_4096_0.json",ref);f.write(a.filename(("calibration",0,"fixed8_native",None,None,None)),row)
            with f.patches():result=a.audit_root(f.root,partial=True)
            s=result["summary"];self.assertTrue(s["pre_freeze_partial"]);self.assertFalse(s["analysis_complete"])
            self.assertIsNone(s["continuation_gate"]);self.assertEqual(s["stages"]["csgb"]["requests"],0)
            self.assertIsNone(s["absolute_metrics"]["csgb"]["tps"]["mean"])
            self.assertIn("PARTIAL",a.report_text(result,[]))
            with f.patches(),self.assertRaises(ValueError):a.audit_root(f.root)

    def test_continuation_requires_feature_matched_constant_and_both_fixed_metrics(self):
        s={"analysis_complete":True,"primary_baseline":"fixed8_native","complete_document_units":list(a.MONITOR),
           "comparisons_all_fixed":{"fixed8_native":{"csgb":{m:{"mean":1.,"ci95":[.1,2.]} for m in ("acceptance","tps")}}},
           "absolute_metrics":{p:{"tps":{"per_document":{str(d):value for d in a.MONITOR}}} for p,value in (("csgb",110.),("constant",100.))}}
        self.assertTrue(a.continuation_gate(s)["passed"])
        s["absolute_metrics"]["constant"]["tps"]["per_document"]["16"]=1000.
        self.assertFalse(a.continuation_gate(s)["passed"])
        s["analysis_complete"]=False;self.assertFalse(a.continuation_gate(s)["evaluated"])


if __name__=="__main__":unittest.main()
