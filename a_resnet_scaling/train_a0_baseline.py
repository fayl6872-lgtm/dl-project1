"""A0: train the ResNet-18-style baseline (2.80M parameters). Only the saved training split is optimized."""
import argparse
import csv
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import torch
from torch.nn import functional as F

from resnet import build_model
from common import ROOT,ImageBank,atomic_json,atomic_torch_save,autocast,evaluate,seed_all,synchronize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/a0_resnet18_baseline.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/a0_baseline')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-batches',type=int,default=0)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    assert config['crop_padding']==4 and config['horizontal_flip_probability']==.5
    assert config['precision']=='bfloat16' and config['scheduler']=='cosine'
    device=torch.device(args.device)
    if args.smoke_batches:
        config['epochs']=2
        assert args.output.name!='a0_baseline','Use a separate output folder for smoke tests.'
    config['smoke_batches']=args.smoke_batches
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=True
    seed_all(config['seed'])
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'history.json').exists() and not args.resume:
        raise RuntimeError('Existing history: use --resume or a fresh output directory.')
    bank=ImageBank(device)
    model=build_model(config).to(device=device,memory_format=torch.channels_last)
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.SGD(model.parameters(),lr=config['learning_rate'],momentum=config['momentum'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'],eta_min=config['minimum_learning_rate'])
    order_rng=torch.Generator(device=device).manual_seed(config['seed'])
    aug_rng=torch.Generator(device=device).manual_seed(config['seed']+1)
    sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('resnet.py','common.py','train_a0_baseline.py')}
    manifest=dict(config=config,parameters=parameters,device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        python=platform.python_version(),versions={x:version(x) for x in ('torch','torchvision','numpy','Pillow')},
        data=bank.manifest,source_sha256=sources,
        initialization='random; no pretrained weights',selection='highest validation accuracy; lower loss breaks ties',
        cudnn_benchmark=True,fully_deterministic=False)
    history=[]
    best_accuracy,best_loss,best_epoch=-1.,float('inf'),0
    first_epoch=1
    if args.resume:
        checkpoint=torch.load(args.output/'last.pt',map_location=device,weights_only=True)
        assert checkpoint['config']==config
        assert checkpoint['source_sha256']==sources,'Training code changed; exact resume refused.'
        assert checkpoint['data_sha256']==bank.manifest['split_sha256']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        history=checkpoint['history']
        best_accuracy,best_loss,best_epoch=checkpoint['best_accuracy'],checkpoint['best_loss'],checkpoint['best_epoch']
        first_epoch=checkpoint['epoch']+1
        order_rng.set_state(checkpoint['order_rng'].cpu())
        aug_rng.set_state(checkpoint['aug_rng'].cpu())
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu(),device)
    atomic_json(args.output/'manifest.json',manifest)
    print(f'PARAMETERS {parameters:,} / {config["parameter_limit"]:,}',flush=True)
    print(f'DATA training={len(bank.train_indices):,} validation={len(bank.val_indices):,}; epochs={config["epochs"]}; device={device}',flush=True)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first_epoch,config['epochs']+1):
        model.train()
        order=bank.train_indices[torch.randperm(len(bank.train_indices),device=device,generator=order_rng)]
        if args.smoke_batches:order=order[:args.smoke_batches*config['batch_size']]
        total_loss=torch.zeros((),device=device)
        correct=torch.zeros((),device=device,dtype=torch.int64)
        lr=optimizer.param_groups[0]['lr']
        synchronize(device);start=time.perf_counter()
        for offset in range(0,len(order),config['batch_size']):
            idx=order[offset:offset+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits=model(bank.batch(idx,augment=True,generator=aug_rng))
                loss=F.cross_entropy(logits,bank.labels[idx])
            loss.backward()
            optimizer.step()
            total_loss+=loss.detach()*len(idx)
            correct+=(logits.detach().argmax(1)==bank.labels[idx]).sum()
        synchronize(device);training_seconds=time.perf_counter()-start
        training_loss=total_loss.item()/len(order)
        if not torch.isfinite(total_loss):raise RuntimeError('Nonfinite loss; stop and inspect training.')
        train_accuracy=100*correct.item()/len(order)
        start=time.perf_counter()
        validation=evaluate(model,bank,bank.val_indices,config['eval_batch_size'])
        synchronize(device);validation_seconds=time.perf_counter()-start
        scheduler.step()
        record=dict(epoch=epoch,learning_rate=lr,train_loss=training_loss,train_accuracy=train_accuracy,
                    val_loss=validation['loss'],val_accuracy=validation['accuracy'],
                    val_correct=validation['correct'],val_total=validation['total'],
                    training_seconds=training_seconds,validation_seconds=validation_seconds)
        history.append(record)
        if validation['accuracy']>best_accuracy or (validation['accuracy']==best_accuracy and validation['loss']<best_loss):
            best_accuracy,best_loss,best_epoch=validation['accuracy'],validation['loss'],epoch
            atomic_torch_save(args.output/'best.pt',dict(model=model.state_dict(),config=config,epoch=epoch,
                validation=validation,parameters=parameters,normalization=dict(mean=bank.manifest['train_mean'],std=bank.manifest['train_std']),
                split_sha256=bank.manifest['split_sha256']))
        atomic_torch_save(args.output/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),config=config,epoch=epoch,history=history,best_accuracy=best_accuracy,
            best_loss=best_loss,best_epoch=best_epoch,order_rng=order_rng.get_state(),aug_rng=aug_rng.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
            source_sha256=sources,data_sha256=bank.manifest['split_sha256']))
        atomic_json(args.output/'history.json',history)
        with (args.output/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0]))
            writer.writeheader();writer.writerows(history)
        atomic_json(args.output/'status.json',dict(completed_epochs=epoch,total_epochs=config['epochs'],best_epoch=best_epoch,best_validation_accuracy=best_accuracy,latest=record))
        print(json.dumps(record),flush=True)
    summary=dict(run_type='smoke' if args.smoke_batches else 'baseline',completed_epochs=len(history),
        trainable_parameters=parameters,parameter_limit=config['parameter_limit'],
        best_epoch=best_epoch,best_validation_accuracy=best_accuracy,best_validation_loss=best_loss,
        final_validation_accuracy=history[-1]['val_accuracy'],
        training_seconds=sum(r['training_seconds'] for r in history),validation_seconds=sum(r['validation_seconds'] for r in history),
        peak_allocated_gpu_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
        test_accuracy=None)
    atomic_json(args.output/'summary.json',summary)
    print('TRAINING COMPLETE '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
