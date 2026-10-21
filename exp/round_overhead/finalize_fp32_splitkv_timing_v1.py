"""Main-agent saved-data cross-check and recoverable read-only experiment seal."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


ROOT = Path(__file__).resolve().parent/'out_fp32_splitkv_timing_20261021_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT/name).read_bytes())


def write(name, value):
    with (ROOT/name).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def main():
    assert not (ROOT/'FINAL_ARTIFACTS_MANIFEST.json').exists()
    summary, audit, comparison = map(read, ('analysis_v2/summary.json', 'audit_v1/AUDIT.json', 'audit_v1/REPORT_COMPARISON.json'))
    assert summary['status'] == 'COMPLETE_MATCHED_DEVELOPMENT_TIMING' and summary['complete_raw_evidence'] and summary['all_outputs_canonical']
    assert summary['issues'] == summary['warnings'] == audit['problems'] == comparison['problems'] == []
    assert audit['status'] == 'AUDIT_PASS' and audit['checks'] == 25891 and audit['all_canonical']
    assert comparison['status'] == 'REPORT_COMPARISON_PASS' and comparison['checks'] == 1293 and comparison['maximum_scalar_absolute_difference'] == 0
    assert not summary['scientific_gate_pass'] and not summary['goal_complete']
    assert not audit['all_six_registered_numeric_gates_with_canonical']
    collection = read('COLLECTION_RECEIPT.json')
    assert collection['scp_exit_codes'] == [0, 0] and collection['collected_files'] == 34
    for name, digest in collection['file_sha256'].items():
        assert sha(ROOT/name) == digest and (ROOT/name).stat().st_size == collection['file_size_bytes'][name]
    for barrier in ('remote_read_only_barrier_before', 'remote_read_only_barrier_after'):
        b = collection[barrier]
        assert b['file_sha256'] == collection['file_sha256'] and b['file_size_bytes'] == collection['file_size_bytes']
        assert b['external_terminal_barrier']['passed'] and b['external_terminal_barrier']['gpu_compute_idle']
    reproducibility = read('REPORT_REPRODUCIBILITY.json')
    assert reproducibility['standalone_exit_code'] == 0 and reproducibility['saved_cli_summary_byte_equal']
    assert reproducibility['tests_passed'] == 12 and reproducibility['tests_skipped'] == 0
    for directory in ('analysis_v1', 'analysis_v2', 'standalone_rerun_v1'):
        assert sha(ROOT/directory/'summary.json') == '4776d5b5f199a879a1ccaa4eea373e3789baf10692eb864b2615e21dc2a4ca35'
    rows = [json.loads(line) for line in (ROOT/'results/RUNTIME.jsonl').read_bytes().splitlines()]
    references = read('results/references.json')
    cold = [json.loads(line) for line in (ROOT/'results/PRIME.jsonl').read_bytes().splitlines()]
    assert len(rows) == 768 and len(cold) == 24
    assert all(r['generated'] == references[f"{r['source_index']}:{r['length']}"] for r in rows+cold)
    docs = [211,213,220,226,233,236,246,248,253,271,272,288,303,306,327,604]
    index = np.random.default_rng(20261031).integers(0,16,size=(10000,16),dtype=np.int64)
    cross_checks = []
    for length in (2048,4096,6144,8192,12288,16384):
        boot, point = {}, {}
        for arm in ('candidate_rescue', 'candidate_retrieval'):
            paired = [[r for r in rows if r['length'] == length and r['arm_id'] == arm and r['source_index'] == d] for d in docs]
            assert all(len(v) == 2 and {r['repeat'] for r in v} == {0,1} for v in paired)
            tokens = np.array([sum(r['tokens'] for r in v) for v in paired], dtype=float)
            seconds = np.array([sum(r['outer_seconds'] for r in v) for v in paired], dtype=float)
            point[arm] = tokens.sum()/seconds.sum()
            boot[arm] = tokens[index].sum(1)/seconds[index].sum(1)
        gain = float(100*(point['candidate_rescue']/point['candidate_retrieval']-1))
        ci = np.percentile(100*(boot['candidate_rescue']/boot['candidate_retrieval']-1),[2.5,97.5])
        expected = summary['statistics']['contexts'][str(length)]['contrasts']['primary_candidate_rescue_vs_candidate_retrieval']
        assert gain == expected['point_percent'] and np.array_equal(ci, expected['ci95_percent'])
        cross_checks.append(dict(length=length, gain_percent=gain, paired_ci95_percent=ci.tolist()))
    visual_names = ('acceptance_length','acceptance_rate','warm_tps','relative_tps_gain','execution_gains','finite_owner_costs','graph_mechanisms','target_scope_calls','round_work')
    assert all((ROOT/'analysis_v2'/(n+ext)).is_file() for n in visual_names for ext in ('.png','.svg'))
    saved = ROOT/'validation_source'
    saved.mkdir()
    for source in (Path(__file__), Path(__file__).with_name('collect_fp32_splitkv_timing_v1.py'), Path(__file__).parents[2]/'tests/test_fp32_splitkv_timing_report_v1.py'):
        shutil.copyfile(source, saved/source.name)
    write('ROOT_REVIEW.json', dict(status='MAIN_AGENT_REVIEW_PASS', actual_evaluated_outputs=792,
          direct_saved_raw_primary_recomputation=cross_checks, goal_complete=False, scientific_gate_pass=False,
          main_agent_rendered_inspection=dict(analysis_v1_all_nine=True, final_v2_display_fixes=['finite_owner_costs','graph_mechanisms'], issues_remaining=[]),
          main_agent_tests=dict(command='python -B -m unittest discover -s tests -p test_fp32_splitkv_timing_report_v1.py -v',passed=12,skipped=0,exit_code=0,elapsed_seconds=15.794,
          provenance='Transcription of main-agent actual tool execution, not an additional run inside this finalizer'),
          source_sha256=sha(Path(__file__)), weights_downloaded=False, remote_mutation=False,
          read_only_protection='Recoverable file mode0444, not operating-system immutable'))
    paths = [p for p in ROOT.rglob('*') if p.is_file()]
    assert not any(p.is_symlink() for p in ROOT.rglob('*'))
    write('FINAL_ARTIFACTS_MANIFEST.json', dict(status='COMPLETE_EVALUATED_DEVELOPMENT_TIMING_ARTIFACTS',
          file_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}))
    write('VERIFIED_COMPLETE.json', dict(status='VERIFIED_MATCHED_DEVELOPMENT_TIMING_COMPLETE',
          completed_at_utc=datetime.now(timezone.utc).isoformat(), raw_calls=792, owners=24, figures_png=9,figures_svg=9,
          report='analysis_v2/REPORT.md',analysis='analysis_v2/analysis_entry.py',raw='results/RUNTIME.jsonl',
          manifest_sha256=sha(ROOT/'FINAL_ARTIFACTS_MANIFEST.json'), scientific_gate_pass=False,goal_complete=False,
          goal_status='ACTIVE_NOT_ACHIEVED',production_speed_claim=False,fresh_confirmation=False,
          all792_original_canonical_tokens=True, preserved_operator_failures=True,rendered_figures_actually_inspected=True,
          protection='Recoverable local Unix file mode0444, not OS immutable'))
    for p in ROOT.rglob('*'):
        if p.is_file():
            p.chmod(0o444)
    manifest = read('FINAL_ARTIFACTS_MANIFEST.json')['file_sha256']
    assert all(sha(ROOT/name) == value and (ROOT/name).stat().st_mode & 0o222 == 0 for name,value in manifest.items())
    print(json.dumps(dict(status='SEALED',files=len(manifest)+2,manifest_sha256=sha(ROOT/'FINAL_ARTIFACTS_MANIFEST.json'),goal_complete=False)))


if __name__ == '__main__':
    main()
