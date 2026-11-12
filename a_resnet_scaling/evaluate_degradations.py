"""Synthetic robustness diagnostics on held-out training-set images only.

These are not hidden-test results or an official corruption benchmark.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur
from common import ImageBank,autocast,atomic_json
from wide_resnet import build_model


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
    device=torch.device('cuda');bank=ImageBank(device)
    cp=torch.load(args.checkpoint,map_location=device,weights_only=True)
    assert not cp['config']['all_data'], 'Only validation-safe ancestors may be evaluated here'
    assert cp['split_sha256']==bank.manifest['split_sha256']
    model=build_model(cp['config']).to(device=device,memory_format=torch.channels_last).eval()
    model.load_state_dict(cp['model']);truth=bank.labels[bank.val_indices].cpu().numpy()
    results={}
    for condition in ['clean','gaussian_noise_003','blur_sigma06','downsample24','brightness07']:
        chunks=[]
        for start in range(0,len(bank.val_indices),256):
            idx=bank.val_indices[start:start+256]
            pixels=bank.images[idx].float()/255
            if condition=='gaussian_noise_003':
                rng=torch.Generator(device=device).manual_seed(44000+start)
                pixels=(pixels+.03*torch.randn(pixels.shape,device=device,generator=rng)).clamp(0,1)
            elif condition=='blur_sigma06':pixels=gaussian_blur(pixels,[3,3],[.6,.6])
            elif condition=='downsample24':
                pixels=F.interpolate(F.interpolate(pixels,size=(24,24),mode='bilinear',align_corners=False,antialias=True),size=(32,32),mode='bilinear',align_corners=False,antialias=True)
            elif condition=='brightness07':pixels=pixels*.7
            x=((pixels-bank.mean)/bank.std).contiguous(memory_format=torch.channels_last)
            with autocast(device):
                p=(model(x).float().softmax(1)+model(x.flip(-1)).float().softmax(1))/2
            chunks.append(p.cpu().numpy())
        probs=np.concatenate(chunks)
        results[condition]=dict(accuracy=float(100*(probs.argmax(1)==truth).mean()),loss=float(-np.log(np.clip(probs[np.arange(len(truth)),truth],1e-12,1)).mean()))
        print(condition,results[condition],flush=True)
    output=dict(checkpoint=str(args.checkpoint.resolve()),examples_per_condition=len(truth),policy='flip',conditions=results,
        note='Synthetic transformations of the same fixed validation images. Conditions are correlated; not independent test sets. No hidden-test data used.')
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,output)


if __name__=='__main__':main()
