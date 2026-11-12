"""Save the raw and EMA weights of a development run's FINAL epoch as separate checkpoints.

Training keeps best.pt (best validation epoch) and last.pt (final raw + EMA weights).
This splits last.pt into runs/<run>/final_raw/best.pt and runs/<run>/final_ema/best.pt so
they can be compared with evaluate_and_export.py. A3's development run continues from
runs/a2_dev_wrn/final_ema/best.pt, and the probe from runs/a3_dev_sam/final_ema/best.pt.

    python extract_final_weights.py --run runs/a2_dev_wrn
"""
import argparse
from pathlib import Path
import torch
from common import ROOT, atomic_torch_save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'runs/a2_dev_wrn')
    run=parser.parse_args().run
    last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    assert last['epoch']==last['config']['epochs'], 'Run has not finished'
    assert not last['config']['all_data']
    best=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    for variant in ['raw','ema']:
        destination=run/f'final_{variant}/best.pt'
        if destination.exists():
            raise FileExistsError(f'Preserve existing candidate: {destination}')
        candidate={key:best[key] for key in ['config','parameters','normalization','split_sha256']}
        for key in ['parent_checkpoint']:
            if key in best:candidate[key]=best[key]
        candidate.update(model=last['model' if variant=='raw' else 'ema'],epoch=last['epoch'],validation=None,variant=variant)
        destination.parent.mkdir(parents=True,exist_ok=True)
        atomic_torch_save(destination,candidate)
        print(destination)


if __name__=='__main__':main()
