"""Display-only legend relocation; the frozen analyzer and summary stay intact."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from unittest.mock import patch

FROZEN_SHA = '4a190f3acec66e58802f83f44b4f97495bae26d8d623aefb0e468f6b320b9453'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path);parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name('analysis_source.py')
    if not source.is_file():
        source = Path(__file__).parent/'out_h100_context_curve_v2_20261020/terminal/tools/report.py'
    assert sha(source) == FROZEN_SHA, 'Frozen analysis source changed'
    spec = importlib.util.spec_from_file_location('_frozen_curve_display_source', source)
    report = importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    original = Axes.legend
    def visible_legend(ax, *values, **kwargs):
        if ax.get_ylabel() == 'Accepted Draft / path tokens\nper verification round':
            kwargs['loc'] = 'lower left'
        return original(ax, *values, **kwargs)
    summary = report.analyze(args.directory)
    with patch.object(Axes, 'legend', visible_legend):
        receipt = report.write_report(summary, args.directory, args.output)
    shutil.copyfile(Path(__file__), args.output/'analysis_entry.py')
    display = dict(status='DISPLAY_ONLY_LEGEND_RELOCATION', frozen_analyzer_sha256=FROZEN_SHA,
        entry_sha256=sha(Path(__file__)), summary_sha256=sha(args.output/'summary.json'),
        change='Only acceptance legend moved to empty lower-left area; no hidden upper CI at16K.',
        statistics_and_validation_unchanged=True)
    with (args.output/'DISPLAY_RECEIPT.json').open('x') as handle:
        json.dump(display, handle, indent=2);handle.write('\n')
    receipt['file_sha256'].update({n:sha(args.output/n) for n in ('analysis_entry.py','DISPLAY_RECEIPT.json')})
    (args.output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=summary['status'], problems=summary['problems'], display=display), indent=2))


if __name__ == '__main__':main()
