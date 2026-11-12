"""Optional BN-buffer recalibration with the *training* augmentation distribution.

Uses allowed training images only. Trainable parameters are unchanged.
"""
import argparse
import hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from wide_resnet import build_model
from randaugment_data import WideTrainingData,EpochSampler
from augment_mixing import cutout
from common import ImageBank,atomic_json,atomic_torch_save,autocast


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda');bank=ImageBank(device)
    cp=torch.load(args.checkpoint,map_location=device,weights_only=True)
    assert not cp['config']['smoke_batches']
    model=build_model(cp['config']).to(device=device,memory_format=torch.channels_last)
    model.load_state_dict(cp['model'])
    ids=torch.arange(len(bank.images),device=device) if cp['config'].get('all_data') else bank.train_indices
    epoch=torch.tensor(1000,dtype=torch.int64).share_memory_()
    dataset=WideTrainingData(ids.cpu().numpy(),bank.manifest,cp['config'],epoch)
    loader=DataLoader(dataset,batch_size=256,sampler=EpochSampler(dataset,809),num_workers=4,pin_memory=True)
    rng=torch.Generator(device=device).manual_seed(809)
    bns=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    momentum=[m.momentum for m in bns]
    for m in bns:m.reset_running_stats()
    model.train();seen=0
    for images,_ in loader:
        x=images.to(device,non_blocking=True,memory_format=torch.channels_last)
        x=cutout(x,cp['config']['cutout_size'],rng)
        for m in bns:m.momentum=len(x)/(seen+len(x))
        with autocast(device):model(x)
        seen+=len(x)
    for m,value in zip(bns,momentum):m.momentum=value
    before=dict(model.named_parameters())
    assert all(torch.equal(p,cp['model'][name]) for name,p in before.items())
    cp['model']=model.state_dict();cp['validation_before_recalibration']=cp['validation'];cp['validation']=None
    cp['bn_recalibration']=dict(training_images=seen,augmentation='same RandAugment/crop/flip/Cutout as training',epoch_seed=1000,order_and_cutout_seed=809,batch_size=256)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_torch_save(args.output,cp)
    atomic_json(args.output.parent/'bn_recalibration.json',dict(source=str(args.checkpoint.resolve()),
        source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),**cp['bn_recalibration']))
    print(f'Recalibrated BN buffers on {seen} allowed training images; all trainable weights unchanged.',flush=True)


if __name__=='__main__':main()
