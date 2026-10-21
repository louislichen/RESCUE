"""Seal COMPLETE horizon evidence locally, only after actual root figure QA.

Never used for a live snapshot, never writes to the remote experiment root,
and never turns a baseline-development result into overall goal completion.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

COUNTS=dict(reference_requests=32,prime_requests=76,timing_requests=2432,owners=76)
FIGURES=('quality','tps','captures','capture_time','retained_tps','cold_service','cumulative','latency','gpu_telemetry')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def file(root,name):
    relative=Path(name);path=root/relative
    need(not relative.is_absolute() and '..' not in relative.parts and path.resolve().is_relative_to(root) and
         path.is_file() and not path.is_symlink(),'Missing/unsafe seal artifact: '+str(name))
    return path


def seal(root,rerun):
    root,rerun=Path(root).resolve(),Path(rerun).resolve()
    outputs=('ANALYSIS_RERUN_VERIFIED.json','FINAL_ARTIFACTS_MANIFEST.json','VERIFIED_COMPLETE.json')
    need(not any((root/n).exists() for n in outputs),'Never overwrite a scientific seal or partial sealing output')
    collection=read(root/'TERMINAL_COLLECTION.json')
    need(collection.get('status')=='TERMINAL_COLLECTED_NOT_YET_INDEPENDENTLY_AUDITED' and collection.get('counts')==COUNTS and
        type(collection.get('pipeline_exit_code')) is int and collection.get('pipeline_exit_code')==0 and all(collection.get(k) is True for k in
        ('read_only','process_and_tmux_absent','complete_file_set_rechecked','all_file_bytes_rechecked')),
         'Terminal handle/byte recheck proof required')
    for name,info in collection['files'].items():
        path=file(root,name);need(sha(path)==info['sha256'] and path.stat().st_size==info['bytes'],'Collected evidence changed: '+name)
    results=root/'results';analysis=results/'analysis_final_v1'
    need(rerun!=analysis.resolve() and rerun.is_dir(),'Standalone rerun must use a distinct completed output directory')
    summary=read(analysis/'summary.json');complete=read(results/'COMPLETE.json');union=read(analysis/'VERIFIED_UNION.json')
    need(summary.get('status')=='COMPLETE' and summary.get('problems')==[] and summary.get('observed_counts')==COUNTS and
        summary.get('authoritative_union') is True and complete.get('passed') is True and complete.get('counts')==COUNTS and
        complete.get('logical_requests')==2540 and complete.get('owner_closed')==76 and complete.get('goal_complete') is False and
        union.get('authoritative') is True and union.get('status')=='VERIFIED_FROZEN_UNION','Strict complete scientific analysis required')
    exit_code=read(root/'PIPELINE_EXIT.json').get('exit_code')
    need(type(exit_code) is int and exit_code==0 and not (root/'FAILURE.json').exists() and not (results/'FAILURE.json').exists(),'Scientific pipeline failed')
    audit=read(root/'INDEPENDENT_RAW_AUDIT.json')
    need(audit.get('status')=='INDEPENDENT_RAW_AUDIT_PASSED' and audit.get('counts')==COUNTS and all(audit.get(k) is True for k in
        ('all2540_requests_verified','all76_owners_verified','retained_union_verified')),'Independent full2540/76/union audit required')
    links=dict(horizon_complete_sha256=results/'COMPLETE.json',horizon_protocol_sha256=results/'protocol.json',
        frozen_selection_sha256=results/'FROZEN_SELECTION.json',analysis_summary_sha256=analysis/'summary.json',
        verified_union_sha256=analysis/'VERIFIED_UNION.json')
    need(all(audit.get(k)==sha(p) for k,p in links.items()),'Independent audit SHA chain differs')
    need(audit.get('audit_source_sha256')==sha(file(root,audit['audit_source_file'])) and
        audit.get('dependency_sha256')==sha(file(root,audit['dependency_file'])),'Independent audit source/dependency changed')
    render_path=analysis/'RENDER_INSPECTION.json';render=read(render_path)
    need(render.get('actually_inspected_by_root') is True and render.get('rendered_pngs_inspected')==list(FIGURES),
         'Actual root inspection of all9 figures required')
    for name in FIGURES:
        for suffix in ('.png','.svg'):
            path=analysis/(name+suffix);need(render.get('figure_sha256',{}).get(path.name)==sha(path),'Inspected figure bytes changed')
    analyzed=read(analysis/'ANALYSIS_COMPLETE.json')
    need(analyzed.get('status')=='COMPLETE' and analyzed.get('artifacts_sha256'),'Complete analysis artifact inventory required')
    for name,digest in analyzed['artifacts_sha256'].items():
        need(sha(file(analysis,name))==digest,'Analysis artifact bytes changed: '+name)
    need((analysis/'summary.json').read_bytes()==(rerun/'summary.json').read_bytes(),
         'Saved standalone source did not exactly reproduce summary')
    need(sha(analysis/'analysis_source.py')==sha(rerun/'analysis_source.py') and
        sha(analysis/'analysis_support.py')==sha(rerun/'analysis_support.py'),
         'Standalone rerun used a different analysis/dependency')
    need(read(rerun/'ANALYSIS_COMPLETE.json').get('status')=='COMPLETE','Standalone full artifact rerun incomplete')
    need((root/'PROGRESS.md').is_file(),'Written interpretation required before sealing')
    rerun_receipt=dict(status='STANDALONE_EXACT_SUMMARY_RERUN_VERIFIED',standalone_summary_exact_rerun=True,
        verified_at_utc=datetime.now(timezone.utc).isoformat(),summary_sha256=sha(analysis/'summary.json'),
        saved_analysis_sha256=sha(analysis/'analysis_source.py'),
        dependency_sha256={'analysis_support.py':sha(analysis/'analysis_support.py')},
        rerun_summary_sha256=sha(rerun/'summary.json'),goal_complete=False)
    # All scientific/visual/rerun checks precede all exclusive seal outputs.
    with (root/outputs[0]).open('x') as handle:json.dump(rerun_receipt,handle,indent=2);handle.write('\n')
    paths=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    need(not any(p.is_symlink() for p in paths),'Refuse symlink targets in final scientific inventory')
    manifest=dict(status='VERIFIED_COMPLETE_HORIZON_DEVELOPMENT_ONLY',goal_complete=False,
        sealed_at_utc=datetime.now(timezone.utc).isoformat(),ignored_non_scientific='__pycache__ directories only',
        file_sha256={p.relative_to(root).as_posix():sha(p) for p in paths})
    with (root/outputs[1]).open('x') as handle:json.dump(manifest,handle,indent=2);handle.write('\n')
    verified=dict(status='VERIFIED_COMPLETE',passed=True,goal_complete=False,
        raw_independent_audit_passed=True,rendered_figures_inspected=True,strict_analysis_complete=True,
        all2540_requests_verified=True,all76_owners_verified=True,retained_union_verified=True,standalone_summary_exact_rerun=True,
        **{k:sha(p) for k,p in links.items()},
        final_artifacts_manifest_sha256=sha(root/outputs[1]),independent_raw_audit_sha256=sha(root/'INDEPENDENT_RAW_AUDIT.json'),
        render_inspection_sha256=sha(render_path),standalone_rerun_receipt_sha256=sha(root/outputs[0]))
    with (root/outputs[2]).open('x') as handle:json.dump(verified,handle,indent=2);handle.write('\n')
    for path in paths+[root/outputs[1],root/outputs[2]]:path.chmod(0o444)
    return verified


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--standalone-rerun',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(seal(args.root,args.standalone_rerun),indent=2))
