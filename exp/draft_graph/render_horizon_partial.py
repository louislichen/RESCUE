"""Add truthful observed-count captions without changing frozen analysis/data.

Invoke with an existing saved PARTIAL analysis directory and a fresh output.
The original scientific analysis and standalone dependency are copied intact;
this separately saved display adapter changes only figure text at save time.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch


def render(saved,output):
    saved,output=Path(saved).resolve(),Path(output).resolve()
    summary=json.loads((saved/'summary.json').read_text())
    if summary.get('status')!='PARTIAL' or output.exists():
        raise RuntimeError('Existing PARTIAL analysis and fresh output required')
    sys.path.insert(0,str(saved))
    spec=importlib.util.spec_from_file_location('frozen_horizon_partial_analysis',saved/'analysis_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    counts=summary['observed_counts']
    observed=(f"PARTIAL | H100/Qwen2: Target FP32, Draft+prefix BF16; exact4096/8192 input, max128/EOS. "
        f"Observed {counts['reference_requests']}/32 AR, {counts['prime_requests']}/76 paid primes, "
        f"{counts['timing_requests']}/2432 warm, {counts['owners']}/76 closed owners.\n")
    cells=[]
    for n,values in summary['metrics'].items():
        for mode,v in values.items():
            cells.append(f"{n}/{mode}:n={v['documents']} paired docs,{v['warm_requests']} paired/{v['actual_raw_warm_requests']} saved warm")
    caption=observed+'; '.join(cells)+'\nQuality/TPS use complete repeat pairs only; capture/latency diagnostics use all saved requests. Blank = unmeasured. Document CIs; not final/held-out/production.'
    original=Figure.savefig
    def savefig(fig,*args,**kwargs):
        if not getattr(fig,'_partial_observed_caption',False):
            for item in fig.texts:
                if item.get_position()[1]<.05:
                    item.set_text(caption);item.set_fontsize(7)
            title=getattr(fig,'_suptitle',None)
            if title is not None:
                text=title.get_text().replace('Primary32-warm TPS points','Observed repeat-paired TPS points').replace('across32 warm requests','across saved warm requests')
                title.set_text('PARTIAL — '+text)
            fig._partial_observed_caption=True
        return original(fig,*args,**kwargs)
    with patch.object(Figure,'savefig',savefig):
        result=module.write_report(summary,saved.parent,output)
    shutil.copyfile(Path(__file__),output/'partial_render_source.py')
    receipt=dict(status='PARTIAL_DISPLAY_ONLY',frozen_summary_unchanged=(saved/'summary.json').read_bytes()==(output/'summary.json').read_bytes(),
        original_analysis_sha256=module.sha(saved/'analysis_source.py'),display_source_sha256=module.sha(Path(__file__)),
        observed_count_caption=caption,figures=result['figures'],scientific_experiment_complete=False)
    with (output/'DISPLAY_RECEIPT.json').open('x') as handle:json.dump(receipt,handle,indent=2);handle.write('\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved_analysis',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(render(args.saved_analysis,args.output),indent=2))
