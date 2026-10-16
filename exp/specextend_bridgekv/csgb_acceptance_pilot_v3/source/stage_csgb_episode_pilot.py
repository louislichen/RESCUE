"""Build one NEW local source capsule; never upload, launch or reuse a root."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

TOP=Path(__file__).resolve().parent
REMOTE=Path('/storage/dzj/spec_repro/CSGB_episode_pilot_20261016_v1')
CAPSULE=TOP/'results/csgb_online_migrated_development_20261016_v1/frozen_source'
PREPARATION=TOP/'results/csgb_episode_pilot_20261016_v1/preparation'
NEW_FILES=(
    'specextend/csgb_episode_policy.py','specextend/csgb_episode_harness.py',
    'specextend/csgb_episode_runner.py','specextend/tests/test_csgb_episode_policy.py',
    'specextend/tests/test_csgb_episode_harness.py','specextend/tests/test_csgb_episode_runner.py',
    'prepare_csgb_episode_scale.py','test_prepare_csgb_episode_scale.py',
    'csgb_episode_gpu_guard.py','test_csgb_episode_gpu_guard.py',
    'analyze_csgb_episode_pilot.py','test_analyze_csgb_episode_pilot.py',
    'stage_csgb_episode_pilot.py','queue_csgb_episode_pilot_v1.sh')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); out=args.out.resolve()
    if out.exists() or out.is_symlink(): raise FileExistsError('Never fill or replace a partial source capsule')
    if out.parent != (TOP/'results/csgb_episode_pilot_20261016_v1').resolve():
        raise ValueError('Use a new capsule under this explicit experiment result directory')
    if sha(CAPSULE/'stage.sha256')!='1efd8a1d2cffea1ca6e756e1b77d8e447507c939f808c9739133be83f11b35bf':
        raise ValueError('Frozen v4 engine stage receipt changed')
    files={}
    for line in (CAPSULE/'stage.sha256').read_text().splitlines():
        wanted,name=line.split(maxsplit=1);name=name.strip()
        if name.startswith('specextend/') and name.endswith('.py') or name=='csgb_migration_provenance.py':
            source=CAPSULE/name
            if sha(source) != wanted: raise ValueError('Changed frozen engine input '+name)
            files[name]=source
    for name in NEW_FILES:
        if name in files: raise ValueError('A new method may not silently overwrite frozen engine '+name)
        source=TOP/name
        if not source.is_file() or source.is_symlink(): raise ValueError('Missing explicit new source '+name)
        files[name]=source
    source=TOP/'results/csgb_online_allround_20261016_v1/frozen_source/train24.jsonl'
    scaler=PREPARATION/'scale_verified.json'
    receipt=TOP/'results/csgb_online_migrated202_20261016_v1/frozen_source/checkpoint_observation.json'
    if sha(receipt)!='62c02665688daea7647ca193cf4c9dcf5029e454caaa624ff267c866cd8b7893':
        raise ValueError('Observed offline checkpoint receipt changed')
    if sha(source)!='1cc4af3d7385f839f7f1e1e484d142d3f945b3d720fa8b465840034fa04d58d6':
        raise ValueError('Only unchanged exact historical training24 corpus may be staged')
    scale=json.loads(scaler.read_text())
    if (scale['source_sha256']!=sha(source) or scale['source_manifest'].get(str(REMOTE/'prepare_csgb_episode_scale.py'))
            !=sha(TOP/'prepare_csgb_episode_scale.py')):
        raise ValueError('Feature scale is not bound to actual new preparation source/corpus')
    files.update({'train24.jsonl':source,'scaler.json':scaler,'checkpoint_observation.json':receipt})
    source_manifest={str(REMOTE/name):sha(path) for name,path in files.items() if name.endswith('.py')}
    protocol={'schema':1,'kind':'csgb_episode_train_only_pilot_v1',
        'fit_documents':list(range(16)),'monitor_documents':list(range(16,24)),'contexts':[4096],
        'development_opened':False,'confirmation_opened':False,'native_v':True,
        'no_automatic_development_or_confirmation_launch':True,'no_epoch_or_seed_selection':True,
        'source_sha256':sha(source),'scaler_sha256':sha(scaler),'checkpoint_receipt_sha256':sha(receipt),
        'checkpoint_training_root':'/storage/dzj/spec_repro/CSGB_online_allround_20261016_v1',
        'hardware':{'physical_gpu_index':4,'expected_gpu_uuid':'GPU-c7642b47-f924-b4b1-3e90-ac94a72f1830'},
        'method':{'budgets':[8,16,24,32],'modes':['csgb','constant','history_native'],
            'seed':101,'order_seed':20261016,'stream_seed':1014096,'temperature':1.,'learning_rate':.0005,
            'passes':2,'initial_weights':'all_zero',
            'gradient':'episode_return_minus_independent_baseline_times_sum_log_policy_scores',
            'round_count_normalization':False,'optimizer':'plain SGD, no clipping/entropy/L2/schedule',
            'learning_rate_choice':'fixed heuristic before real pilot outcomes; not tuned on monitor',
            'training_and_deployment':'same categorical softmax sampling, parameters fixed within every request',
            'baseline':'per-fit-document NativeFixed8 N-rho*T from pretraining complete-request calibration',
            'rho':'sum fit NativeFixed8 usefulN / sum fit NativeFixed8 whole-request T; frozen before training'},
        'monitor':{'policies':['fixed8_native','fixed16_native','fixed24_native','fixed32_native',
                               'csgb','constant','history_native'],
            'action_streams':3,'timing_repeats':2,'warmups_per_policy':1,
            'baseline_selection':'four fixed Native arms fit0..15 only; max equal-document meanTPS; tie smaller budget',
            'no_model_updates':True,'no_epoch_or_seed_selection':True,
            'aggregation':'per action stream median2 timing repeats, then equal average over3 streams within document',
            'stochastic_uncertainty':'3 independent action streams, not6 independent documents; paired document n8'},
        'settings':{'model_seed':42,'target_dtype':'float32','draft_dtype':'float16','max_new_tokens':128,
            'max_depth':10,'threshold':.7,'cache_refresh_every':4,'chunk_size':32,'retrieve_top_k':32,
            'corrected_root_coverage':True,'terminal_draft_elision':True,'tf32':False},
        'source_manifest':source_manifest,
        'analysis':{'document_bootstrap_draws':10000,'bootstrap_seed':42,'paired_unit':'document',
            'primary_reference':'fit-selected best fixed Native, never monitor-selected',
            'primary_metrics':['scored Draft acceptance per verified round','complete-request returned tokens/seconds'],
            'continuation_gate':'4K CSGB vs fit-selected best fixed: acceptance and TPS mean>0 and paired95CI lower>0; CSGB vs featurematchedconstant TPS mean>0 and lower>0',
            'partial_results_cannot_pass':True,'confirmatory_claim':False,'production_speed_claim':False,
            'positive_pilot_is_not_goal_completion':True},
        'disclosure':['actual4096-only feasibility, not8K/long-context success',
            'historical train24 reused; fit16/internal monitor8 not globally unseen data',
            'one policy-training seed, final pass2 only; no hyperparameter or winner search',
            'server48GB-class RTX4090; eagerFP32Target/FP16Draft; explicit request-boundary sync; implicit waits remain',
            'complete-request optimization differs from macro document medianTPS gate; passing must be measured',
            'no previous synchronized round timing/rho is reused; scale uses only699 audited fit0..15 features'],
        'next_stage':'Only if justified: separately preregister both4K/8K, multiple seeds, development24..31 and untouched confirmation32..47; no automatic launch'}
    out.mkdir()
    for name,path in files.items():
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    with (out/'pilot_protocol.json').open('x') as handle:
        json.dump(protocol,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
    manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    with (out/'stage.sha256').open('x') as handle:
        for name,digest in sorted(manifest.items()):handle.write(digest+'  '+name+'\n')
    print(json.dumps({'capsule':str(out),'files':len(manifest),'remote_root':str(REMOTE),
        'stage_sha256':sha(out/'stage.sha256'),'protocol_sha256':sha(out/'pilot_protocol.json'),
        'uploaded':False,'launched':False},sort_keys=True))


if __name__=='__main__': main()
