"""Frozen development report with narrowly verified old-metadata relocation.

This new CPU-only entry never changes saved protocols, recorded measurements,
the frozen report, or its four dependencies. Only three old cohort metadata
names may acquire an in-memory byte alias for the stage2 asset comparison.
Models, tokenizer, paid prime, Mapper, policy and cost have no alias rule.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

FROZEN_SHA256 = {
    'analysis_source.py': '9b5d4a2259569b5783e8e241b7f3dfce3fc2ef96d034592bef68c7d8d7c82467',
    'analysis_support.py': '2c2885f4137f2cb772e3f157568f7a38f0ff9162942adc645c026abcbbbf252e',
    'analysis_horizon_support.py': '3ee330dbcd3e82b3c4c4661105d2fce3cab64c8a1631de6f883db070683b5d48',
    'analysis_fair_support.py': '3172d378f55e0464ca6b2349ec0fb60747f588607b30e508f69e70a5db5bb63f',
    'analysis_union_support.py': '8a8a8d32ab1e47a26cc7a5fde8ddf8fdb56921df99b4270c1ef709288f9cce7b',
}
REPO_SOURCES = {
    'analysis_source.py': 'exp/draft_graph/report_frozen_devcheck.py',
    'analysis_support.py': 'rescue/baseline_selection_report.py',
    'analysis_horizon_support.py': 'rescue/baseline_horizon_report.py',
    'analysis_fair_support.py': 'rescue/fair_confirmation_report.py',
    'analysis_union_support.py': 'rescue/fair_union_confirmation_report.py',
}
OLD_ROOT = '/home/wanghl/cl/RESCUE/'
ALLOWED_METADATA = (
    'exp/long_system/data/train.jsonl',
    'exp/module_opt/data_v1_20261018/confirmation.jsonl',
    'exp/module_arch/data_v1_20261018/confirmation.jsonl',
)
ASSET_PROBLEM = 'Stage2: Model/tokenizer/old-prime assets differ from study'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_frozen():
    here = Path(__file__).resolve().parent
    paths = {}
    for saved, relative in REPO_SOURCES.items():
        path = here / saved
        if not path.is_file():
            path = here.parents[1] / relative
        if not path.is_file() or sha(path) != FROZEN_SHA256[saved]:
            raise ValueError('Frozen dependency bytes changed or absent: ' + saved)
        paths[saved] = path
    # The unmodified report itself imports its saved siblings or repo supports.
    spec = importlib.util.spec_from_file_location('_relocated_frozen_devcheck', paths['analysis_source.py'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for saved, path in module.SUPPORT_PATHS.items():
        if sha(path) != FROZEN_SHA256[saved]:
            raise ValueError('Actual imported frozen support differs: ' + saved)
    return module


def _safe_relative(value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Invalid archive-relative path')
    if Path(value).as_posix() != value:
        raise ValueError('Noncanonical archive-relative path')
    return value


def verified_aliases(evidence, protocol, frozen):
    """Authenticate all byte links before any old-name alias can be used.

    Original absolute names are labels, NEVER paths to open. Every opened
    payload is resolved below the supplied portable evidence tree.
    """
    entry = protocol.get('stage2_evidence', {})
    if entry.get('root') != 'stage2_evidence':
        raise ValueError('Only the portable stage2 evidence tree may relocate metadata')
    base = frozen.f.Evidence(evidence.path('stage2_evidence'))
    result = _safe_relative(entry.get('baseline_results'))
    manifest_path = _safe_relative(entry.get('final_manifest'))
    final = base.json(manifest_path)
    inventory = final.get('file_sha256', {})
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('Sealed stage2 inventory missing')
    old_path = result + '/protocol.json'
    old_file = base.path(old_path)
    if inventory.get(old_path) != sha(old_file):
        raise ValueError('Sealed stage2 protocol SHA mismatch')
    old = base.json(old_path)
    old_assets, current = old.get('assets_sha256', {}), protocol.get('assets_sha256', {})
    old_paths = old.get('data_source_paths', {})
    own_paths = protocol.get('data_source_paths', {})
    if own_paths != old_paths:
        raise ValueError('Own and sealed stage2 metadata layouts differ')
    old_manifest_name = result + '/' + _safe_relative(old.get('data_manifest'))
    old_manifest_file = base.path(old_manifest_name)
    old_manifest_digest = sha(old_manifest_file)
    if (inventory.get(old_manifest_name) != old_manifest_digest or
            old.get('data_manifest_sha256') != old_manifest_digest):
        raise ValueError('Sealed stage2 data manifest SHA mismatch')
    old_manifest = base.json(old_manifest_name)
    metadata_sha = old_manifest.get('source_sha256', {})
    own_stage2 = protocol.get('args', {}).get('stage2_results')
    if not isinstance(own_stage2, str) or not Path(own_stage2).is_absolute() or '..' in Path(own_stage2).parts:
        raise ValueError('Frozen stage2-results label invalid')
    aliases = {}
    records = []
    for name in ALLOWED_METADATA:
        original = OLD_ROOT + name
        digest = old_assets.get(original)
        relative = 'data/' + name
        if (not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or
                old_paths.get(name) != relative or own_paths.get(name) != relative):
            raise ValueError('Fixed metadata name/layout/digest missing: ' + name)
        remote_label = str(Path(own_stage2) / relative)
        local_path = evidence.path(relative)
        sealed_name = result + '/' + relative
        sealed_path = base.path(sealed_name)
        if (current.get(remote_label) != digest or metadata_sha.get(name) != digest or
                inventory.get(sealed_name) != digest or sha(local_path) != digest or sha(sealed_path) != digest):
            raise ValueError('Metadata relocation byte/SHA binding mismatch: ' + name)
        # Existing original names may not be changed, hidden, or repaired.
        if original in current and current[original] != digest:
            raise ValueError('Existing original metadata asset has wrong SHA: ' + name)
        needed = original not in current
        if needed:
            aliases[original] = digest
        records.append(dict(metadata_name=name, original_asset_label=original,
            frozen_relocated_asset_label=remote_label, local_saved_path=relative,
            sealed_saved_path='stage2_evidence/' + sealed_name, sha256=digest,
            alias_needed=needed, sealed_manifest_sha256=sha(base.path(manifest_path)),
            sealed_protocol_sha256=sha(old_file), sealed_data_manifest_sha256=old_manifest_digest))
    if base.problems:
        raise ValueError('Relocation evidence read failed: ' + '; '.join(base.problems))
    return aliases, records


def analyze(directory, frozen=None):
    frozen = _load_frozen() if frozen is None else frozen
    original = frozen.analyze(directory)
    old_portable = frozen.f._portable_study
    receipts, errors = [], []

    def portable(evidence, protocol, key, kind):
        if kind != 2:
            return old_portable(evidence, protocol, key, kind)
        try:
            aliases, records = verified_aliases(evidence, protocol, frozen)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            problem = 'Verified metadata relocation failed: ' + str(error)
            errors.append(problem)
            evidence.problem(problem)
            return old_portable(evidence, protocol, key, kind)
        receipts.extend(records)
        # Only this one helper call sees the verified labels. The raw protocol,
        # original asset mapping, other checks and stage1 are never altered.
        adjusted = dict(protocol, assets_sha256=dict(protocol.get('assets_sha256', {}), **aliases))
        return old_portable(evidence, adjusted, key, kind)

    frozen.f._portable_study = portable
    try:
        result = frozen.analyze(directory)
    finally:
        frozen.f._portable_study = old_portable
    # Only proof/status consequences can differ. No numeric statistic, count,
    # recorded diagnostic, opponent selection or input cell may move.
    exempt = {'status', 'problems', 'development_feasible'}
    for key in set(original) | set(result):
        if key not in exempt and original.get(key) != result.get(key):
            raise ValueError('Relocation changed a frozen measurement field: ' + key)
    removed = [p for p in original.get('problems', []) if p not in result.get('problems', [])]
    if any(problem != ASSET_PROBLEM for problem in removed):
        raise ValueError('Relocation removed an unrelated original evidence problem')
    result['relocation_receipt'] = dict(schema=1,
        status='VERIFIED_METADATA_RELOCATION' if not errors and len(receipts) == 3 else 'RELOCATION_FAILED',
        only_stage2=True, metadata_only=True, original_protocol_unchanged=True,
        original_absolute_files_opened=False, aliases=receipts, problems=errors,
        original_analysis_status=original.get('status'), original_analysis_problems=original.get('problems', []),
        removed_problems=removed, frozen_measurement_fields_identical=True,
        frozen_source_sha256=FROZEN_SHA256,
        scope='Three fixed metadata byte aliases only; no model/tokenizer/prime/Mapper/policy/cost aliases. No measurement, statistics, choice, count or scientific protocol change.')
    return result


def write_report(summary, directory, output, frozen=None):
    frozen = _load_frozen() if frozen is None else frozen
    output = Path(output)
    # Frozen writer retains its exact source and four original supports.
    frozen.write_report(summary, directory, output)
    shutil.copyfile(Path(__file__), output / 'analysis_entry.py')
    report = output / 'REPORT.md'
    text = report.read_text()
    text += ('\n## Verified archive relocation\n\n'
        'Run [analysis_entry.py](analysis_entry.py), not analysis_source.py, to reproduce this report. '
        'The latter and all four original supports remain byte-identical frozen dependencies. '
        'Only the three documented old cohort metadata names may acquire verified in-memory SHA aliases '
        'after their original label, frozen relocated label, local bytes, sealed bytes, data manifest and '
        'sealed artifact inventory agree. Model/tokenizer/prime/Mapper/policy/cost comparison stays strict. '
        'No raw record or protocol is modified. Every measurement/count/choice/diagnostic matches the '
        'original report; only evidence validation and its status/go-no-go consequences may change. '
        'The deterministic relocation_receipt is saved in summary.json. This is not an independent audit.\n')
    report.write_text(text)
    receipt = dict(status=summary.get('status', 'PARTIAL'), goal_complete=False,
        fresh_confirmation=False, render_required=True, rendered_figures=list(frozen.FIGURES),
        file_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'ANALYSIS_COMPLETE.json'},
        analysis_entry='analysis_entry.py', frozen_dependencies_sha256=FROZEN_SHA256,
        independent_audit=False,
        scope='Development-only relocated metadata verification; actual root figure inspection still required')
    (output / 'ANALYSIS_COMPLETE.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh analysis output required; never overwrite')
    frozen = _load_frozen()
    summary = analyze(args.directory, frozen)
    write_report(summary, args.directory, args.output, frozen)
    print(json.dumps(dict(status=summary.get('status'), goal_complete=False,
        development_feasible=summary.get('development_feasible'), problems=summary.get('problems'),
        relocation_status=summary['relocation_receipt']['status']), indent=2))


if __name__ == '__main__':
    main()
