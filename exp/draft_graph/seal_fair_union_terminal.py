"""Seal complete UNION evidence locally, including negative speed results.

Requires actual root render inspection, independent raw/point auditing and an
exact saved-source rerun. This tool never decides that the user's goal is done.
It writes only three new exclusive seal files and makes saved artifacts readonly.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

FIGURES=('acceptance','tps','cold_service','latency_captures','actions_selection')
ANALYSIS_SOURCES={'analysis_source.py':'rescue/fair_union_confirmation_report.py',
    'analysis_fair_support.py':'rescue/fair_confirmation_report.py',
    'analysis_horizon_support.py':'rescue/baseline_horizon_report.py',
    'analysis_support.py':'rescue/baseline_selection_report.py'}
AUDIT_FLAGS=('passed','all_requests_verified','all_owners_verified','all_quality_owners_verified','all_timing_owners_verified',
    'all_canonical','own_quality_native_query_equal','nonzero_mapper','baseline_quality_checked','raw_tree_verified',
    'retained_union_verified','source_assets_verified','report_points_verified')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file(root,name):
    need(isinstance(name,str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe saved seal path')
    path=root/name
    need(path.resolve().is_relative_to(root.resolve()) and path.is_file() and
        not any(p.is_symlink() for p in (path,*path.parents) if p.is_relative_to(root)),'Missing/unsafe seal artifact: '+name)
    return path


def read(path):
    value=json.loads(Path(path).read_text());need(isinstance(value,dict),'Expected JSON object: '+str(path));return value


def collector_module():
    path=collector_source()
    spec=importlib.util.spec_from_file_location('sealed_union_collection_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def collector_source():
    path=Path(__file__).with_name('collect_fair_union_terminal.py')
    if not path.exists():path=Path(__file__).with_name('SEALER_SUPPORT.py')
    need(path.is_file() and not path.is_symlink(),'Missing/unsafe standalone sealer support source')
    return path


def inventory(root,contract):
    """No external provenance path is followed, even by the hash barrier."""
    paths=[]
    for path in root.rglob('*'):
        need(not path.is_symlink(),'Refuse any final symlink target')
        if '__pycache__' in path.parts:continue
        if path.is_file():
            need(not contract.FORBIDDEN.intersection(path.relative_to(root).parts) and
                (path.suffix.lower() in contract.SUFFIXES or path.name in contract.SPECIAL),'Unregistered final binary/model/corpus artifact')
            file(root,path.relative_to(root).as_posix());paths.append(path)
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(paths)}


def seal(root,rerun,analysis=None):
    root,rerun=Path(root).resolve(),Path(rerun).resolve()
    analysis=(root/'results/analysis_final_v1' if analysis is None else Path(analysis)).resolve()
    need(root.is_dir() and not root.is_symlink() and analysis.is_relative_to(root) and analysis.is_dir(),'Missing local complete root/analysis')
    outputs=('ANALYSIS_RERUN_VERIFIED.json','FINAL_ARTIFACTS_MANIFEST.json','VERIFIED_COMPLETE.json')
    own_sources=('SEALER_SOURCE.py','SEALER_SUPPORT.py')
    need(not any((root/name).exists() for name in (*outputs,*own_sources)),'Never overwrite a seal or partial seal output')
    source_bytes=(Path(__file__).read_bytes(),collector_source().read_bytes())
    collection=read(file(root,'TERMINAL_COLLECTION.json'));protocol=read(file(root,'results/protocol.json'));contract=collector_module()
    before=inventory(root,contract)
    m,r,expected,total=contract.matrix(protocol)
    need(collection.get('status')=='UNION_TERMINAL_COLLECTED_NOT_YET_INDEPENDENTLY_AUDITED' and collection.get('M')==m and collection.get('R')==r and
        collection.get('counts')==expected and collection.get('expected_counts')==expected and collection.get('logical_requests')==total and
        collection.get('expected_requests')==total and type(collection.get('pipeline_exit_code')) is int and collection['pipeline_exit_code']==0 and all(
        collection.get(k) is True for k in ('read_only','process_and_tmux_absent','complete_file_set_rechecked','all_file_bytes_rechecked','snapshot_is_atomic')),
        'Complete terminal handle/file barriers and dynamic union matrix required')
    for key in ('handles_before','handles_after'):
        state=collection.get(key,{})
        need(state.get('tmux_present') is False and state.get('actual_python_processes')==[] and state.get('actual_execution_processes')==[],
            'Actual terminal process/tmux proof missing')
    collected=collection.get('files',{});need(collected,'Empty terminal collection inventory')
    for name,info in collected.items():
        path=file(root,name);need(path.stat().st_size==info.get('bytes') and sha(path)==info.get('sha256'),'Collected bytes changed: '+name)
    need(collection.get('collector_source_sha256')==sha(file(root,'COLLECTOR_SOURCE.py')),'Actual collector source receipt changed')
    exported=read(file(root,'EVIDENCE_MANIFEST.json')).get('file_sha256',{})
    need(exported and exported.get('TERMINAL_COLLECTION.json')==sha(root/'TERMINAL_COLLECTION.json'),'Export manifest does not bind collection receipt')
    for name,digest in exported.items():need(sha(file(root,name))==digest,'Exported evidence changed: '+name)
    payloads={name:file(root,name).read_bytes() for name in collected}
    contract.execution_chain(payloads,Path(collection['root']),protocol,True)
    need(all(not (root/name).exists() for name in ('FAILURE.json','results/FAILURE.json','data_gate/GATE_FAILURE.json','data_gate/data/DATA_PREP_FAILURE.json')),
        'Failed/partial execution cannot be sealed complete')
    done=read(file(root,'results/COMPLETE.json'))
    need(done.get('passed') is True and done.get('counts')==expected and done.get('expected_counts')==expected and done.get('logical_requests')==total and
        done.get('comparisons')==m and done.get('matching_geometry_count')==r and done.get('goal_complete') is False and all(done.get(k) is True for k in (
        'sources_unchanged','assets_unchanged','dependencies_unchanged','quality_gate_passed','all_canonical','owners_closed','analysis_required_locally','data_unchanged','dual_study_freeze_unchanged')),
        'Complete scientific execution/exactness/immutability receipt required')
    summary=read(file(analysis,'summary.json'));stats=summary.get('statistics',{})
    need(summary.get('status')=='COMPLETE' and summary.get('problems')==[] and summary.get('comparisons')==m and
        summary.get('observed_counts')==expected and summary.get('expected_counts')==expected and summary.get('expected_requests')==total and
        summary.get('observed_requests')==total and stats.get('n')==16 and stats.get('comparisons')==m and stats.get('per_context_n')=={'4096':16,'8192':16} and
        stats.get('draws')==100000 and stats.get('seed')==20261026 and stats.get('simultaneous_lower_quantile')==.05/m and
        stats.get('document_index_shared_across_all_contrasts') is True,'Complete frozen whole-document analysis required')
    contrasts=stats.get('contrasts',[])
    expected_contrasts={(int(n),mode) for n,values in protocol['frozen_choices'].items() for mode in values}
    need(len(contrasts)==m and {(c.get('length'),c.get('baseline_mode')) for c in contrasts}==expected_contrasts,'All retained contrast effects required')
    need(all(type(c.get(k)) in (int,float) and math.isfinite(c[k]) for c in contrasts for k in
        ('point_percent','simultaneous_one_sided_lower95')),'Non-finite saved contrast effect/bound')
    numerical_gate=all(c['point_percent']>0 and c['simultaneous_one_sided_lower95']>0 for c in contrasts)
    need(type(stats.get('statistical_gate')) is bool and stats['statistical_gate'] is numerical_gate and
        type(summary.get('verified_success')) is bool and summary['verified_success'] is numerical_gate,'Reported gate differs from all-M saved effect/bound signs')
    analyzed=read(file(analysis,'ANALYSIS_COMPLETE.json'));pins=analyzed.get('file_sha256',{})
    need(analyzed.get('status')=='COMPLETE' and analyzed.get('verified_success') is numerical_gate and analyzed.get('render_required') is True and
        analyzed.get('rendered_figures')==list(FIGURES) and pins,'Complete rendered analysis inventory required')
    for name,digest in pins.items():need(sha(file(analysis,name))==digest,'Analysis artifact changed: '+name)
    for name,source in ANALYSIS_SOURCES.items():
        need(sha(file(analysis,name))==protocol.get('source_sha256',{}).get(source),'Saved standalone source differs from executed frozen source: '+name)
    render=read(file(analysis,'RENDER_INSPECTION.json'))
    need(render.get('actually_inspected_by_root') is True and render.get('rendered_pngs_inspected')==list(FIGURES),'Actual root inspection of all five figure families required')
    for name in FIGURES:
        for suffix in ('.png','.svg'):
            need(render.get('figure_sha256',{}).get(name+suffix)==sha(file(analysis,name+suffix)),'Inspected figure changed: '+name+suffix)
    audit=read(file(root,'INDEPENDENT_UNION_RAW_AUDIT.json'))
    need(audit.get('status')=='INDEPENDENT_UNION_RAW_AUDIT_PASSED' and all(audit.get(k) is True for k in AUDIT_FLAGS) and
        audit.get('M')==m and audit.get('R')==r and audit.get('counts')==expected and audit.get('expected_counts')==expected and audit.get('logical_requests')==total and
        audit.get('independent_ci_verified') is False,'Independent complete raw/quality/point audit required; do not invent independent CI verification')
    links=dict(protocol_sha256=root/'results/protocol.json',complete_sha256=root/'results/COMPLETE.json',analysis_summary_sha256=analysis/'summary.json',
        collection_manifest_sha256=root/'TERMINAL_COLLECTION.json')
    need(all(audit.get(k)==sha(path) for k,path in links.items()),'Independent raw audit SHA links differ')
    need(audit.get('audit_source_file')=='INDEPENDENT_UNION_RAW_AUDIT_SOURCE.py' and audit.get('dependency_file')=='INDEPENDENT_UNION_RAW_AUDIT_SUPPORT.py' and
        audit.get('audit_source_sha256')==sha(file(root,audit['audit_source_file'])) and audit.get('dependency_sha256')==sha(file(root,audit['dependency_file'])),
        'Independent raw audit source/dependency bytes changed')
    need(rerun!=analysis and rerun.is_dir(),'A distinct completed standalone rerun directory required')
    need(file(analysis,'summary.json').read_bytes()==file(rerun,'summary.json').read_bytes(),'Saved standalone source did not exactly reproduce summary bytes')
    for name in ANALYSIS_SOURCES:need(sha(file(analysis,name))==sha(file(rerun,name)),'Standalone rerun used another source/dependency: '+name)
    rerun_complete=read(file(rerun,'ANALYSIS_COMPLETE.json'))
    need(rerun_complete.get('status')=='COMPLETE' and rerun_complete.get('verified_success') is numerical_gate and rerun_complete.get('rendered_figures')==list(FIGURES),
        'Standalone full artifact rerun incomplete')
    for name,digest in rerun_complete.get('file_sha256',{}).items():need(sha(file(rerun,name))==digest,'Standalone rerun artifact changed: '+name)
    need(rerun_complete.get('file_sha256'),'Standalone rerun artifact pins missing')
    need(file(root,'PROGRESS.md').stat().st_size>0 and file(analysis,'REPORT.md').stat().st_size>0,'Written bounded-scope interpretation/report required')
    # Validate every final saved path BEFORE creating any seal output. Saved
    # artefacts may include independent audit/rerun additions after collection,
    # but never credential/model/corpus files or symlink dereferences.
    need(inventory(root,contract)==before,'Final saved file set/bytes changed during sealing validation')
    paths=[root/name for name in before]
    rerun_receipt=dict(schema=1,status='STANDALONE_EXACT_SUMMARY_RERUN_VERIFIED',standalone_summary_exact_rerun=True,
        verified_at_utc=datetime.now(timezone.utc).isoformat(),summary_sha256=sha(analysis/'summary.json'),
        saved_analysis_sha256=sha(analysis/'analysis_source.py'),dependency_sha256={n:sha(analysis/n) for n in ANALYSIS_SOURCES if n!='analysis_source.py'},
        rerun_summary_sha256=sha(rerun/'summary.json'),scientific_speed_gate_passed=numerical_gate,goal_complete=False)
    need(source_bytes==(Path(__file__).read_bytes(),collector_source().read_bytes()),'Sealing source changed during validation')
    for name,raw in zip(own_sources,source_bytes):
        with (root/name).open('xb') as handle:handle.write(raw)
        paths.append(root/name)
    with (root/outputs[0]).open('x') as handle:json.dump(rerun_receipt,handle,indent=2);handle.write('\n')
    paths.append(root/outputs[0])
    manifest=dict(schema=1,status='VERIFIED_COMPLETE_UNION_CONFIRMATION_EVIDENCE',goal_complete=False,
        scientific_speed_gate_passed=numerical_gate,sealed_at_utc=datetime.now(timezone.utc).isoformat(),
        ignored_non_scientific='__pycache__ directories only',file_sha256={p.relative_to(root).as_posix():sha(p) for p in sorted(paths)})
    with (root/outputs[1]).open('x') as handle:json.dump(manifest,handle,indent=2);handle.write('\n')
    verified=dict(schema=1,status='VERIFIED_COMPLETE',passed=True,goal_complete=False,scientific_speed_gate_passed=numerical_gate,
        goal_decision='Reserved for the root after verified handoff; an evidence seal never automatically completes the user goal',
        M=m,R=r,counts=expected,logical_requests=total,all_requests_verified=True,all_timing_owners_verified=True,all_quality_owners_verified=True,
        raw_independent_audit_passed=True,rendered_figures_inspected=True,strict_analysis_complete=True,standalone_summary_exact_rerun=True,
        independent_ci_verified=False,**{k:sha(path) for k,path in links.items()},
        sealer_source_sha256=sha(root/own_sources[0]),sealer_dependency_sha256=sha(root/own_sources[1]),
        final_artifacts_manifest_sha256=sha(root/outputs[1]),independent_raw_audit_sha256=sha(root/'INDEPENDENT_UNION_RAW_AUDIT.json'),
        render_inspection_sha256=sha(analysis/'RENDER_INSPECTION.json'),standalone_rerun_receipt_sha256=sha(root/outputs[0]))
    with (root/outputs[2]).open('x') as handle:json.dump(verified,handle,indent=2);handle.write('\n')
    for path in paths+[root/outputs[1],root/outputs[2]]:path.chmod(0o444)
    return verified


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--standalone-rerun',type=Path,required=True);parser.add_argument('--analysis',type=Path)
    args=parser.parse_args();print(json.dumps(seal(args.root,args.standalone_rerun,args.analysis),indent=2))


if __name__=='__main__':main()
