"""Robustness probe (report Table 5): image-degradation continuation of an internally trained Wide ResNet; no external weights.

All-data training (--all-data) never reports
validation accuracy: the original validation images are then training examples.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from augment_mixing import cutout
from augment_degradation import degradation_augment
from wide_resnet import build_model
from randaugment_data import WideTrainingData, EpochSampler
from torch.utils.data import DataLoader
from torch.nn import functional as F
from common import ROOT, ImageBank, atomic_json, atomic_torch_save, autocast, evaluate, seed_all, synchronize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-data', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-batches', type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(smoke_batches=args.smoke_batches, all_data=args.all_data)
    if args.smoke_batches:
        config.update(epochs=2, warmup_epochs=0, mix_off_last_epochs=0)
    assert config['precision'] == 'bfloat16'
    assert config['crop_padding'] == 4 and config['horizontal_flip_probability'] == .5
    assert torch.cuda.is_available(), 'This experiment expects the local CUDA GPU.'
    device = torch.device('cuda')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    seed_all(config['seed'])
    bank = ImageBank(device)
    ids = torch.arange(len(bank.images), device=device) if args.all_data else bank.train_indices
    model = build_model(config).to(device=device, memory_format=torch.channels_last)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count <= 5_000_000
    parent_path=ROOT/config['initial_checkpoint']
    parent=torch.load(parent_path,map_location=device,weights_only=True)
    assert parent['config']['all_data']==args.all_data, 'Prevent validation leakage from all-data ancestor'
    assert parent['split_sha256']==bank.manifest['split_sha256']
    assert parent['parameters']==count and not parent['config']['smoke_batches']
    model.load_state_dict(parent['model'])
    provenance=dict(path=str(parent_path.resolve()),sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                    epoch=parent['epoch'],cumulative_epochs=parent['epoch']+parent.get('parent_checkpoint',{}).get('cumulative_epochs',parent.get('parent_checkpoint',{}).get('epoch',0)),origin='own model trained from random initialization on permitted data only')
    # EMA is an alternative set of weights of the SAME architecture, not an ensemble.
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=.9,
                                nesterov=True, weight_decay=config['weight_decay'])
    order_rng = torch.Generator().manual_seed(config['seed'])
    epoch_shared=torch.zeros((),dtype=torch.int64).share_memory_()
    dataset=WideTrainingData(ids.cpu().numpy(),bank.manifest,config,epoch_shared)
    loader=DataLoader(dataset,batch_size=config['batch_size'],sampler=EpochSampler(dataset,config['seed']),
                      num_workers=config['num_workers'],pin_memory=True,persistent_workers=config['num_workers']>0,
                      generator=order_rng)
    aug_rng = torch.Generator(device=device).manual_seed(config['seed']+1)
    mix_rng = np.random.default_rng(config['seed']+2)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'history.json').exists() and not args.resume:
        raise RuntimeError('Existing run; use --resume or a new directory.')
    source_names = ('resnet.py','wide_resnet.py','randaugment_data.py','common.py','augment_mixing.py','augment_degradation.py','train_degradation_probe.py')
    sources = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names}
    history, best, first, steps = [], (-1., float('inf')), 1, 0
    if args.resume:
        c = torch.load(args.output/'last.pt', map_location=device, weights_only=True)
        assert c['config'] == config and c['source_sha256'] == sources
        assert c['split_sha256'] == bank.manifest['split_sha256']
        model.load_state_dict(c['model']); ema.load_state_dict(c['ema'])
        optimizer.load_state_dict(c['optimizer'])
        history, best, first, steps = c['history'], tuple(c['best']), c['epoch']+1, c['steps']
        order_rng.set_state(c['order_rng'].cpu()); aug_rng.set_state(c['aug_rng'].cpu())
        mix_rng.bit_generator.state = c['mix_rng']
        torch.set_rng_state(c['torch_rng'].cpu()); torch.cuda.set_rng_state(c['cuda_rng'].cpu())
    manifest = dict(config=config, parameters=count, source_sha256=sources, data=bank.manifest,
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__, python=platform.python_version(),
                    training_images=len(ids), validation_images=0 if args.all_data else len(bank.val_indices),
                    normalization='fixed training-subset statistics (also retained for final all-data run)',
                    initialization='continued from own scratch-trained checkpoint; no external data or pretrained weights',parent_checkpoint=provenance,
                    selection='fixed final epoch and weight variant' if args.all_data else 'validation accuracy; loss breaks ties',
                    augmentation='training-only stochastic noise, resampling, blur or brightness on 35% of images; one SGD pass',ema='same architecture, averaged parameters; current batch-norm buffers; no prediction ensemble')
    atomic_json(args.output/'manifest.json', manifest)
    # Snapshot source, so later edits cannot erase the recipe behind these results.
    source_dir=args.output/'source'; source_dir.mkdir(exist_ok=True)
    for name in source_names:
        (source_dir/name).write_bytes((ROOT/name).read_bytes())
    def save_model(path, network, epoch, metrics, variant):
        atomic_torch_save(path, dict(model=network.state_dict(), config=config, epoch=epoch,
            validation=metrics, parameters=count, variant=variant,parent_checkpoint=provenance,
            normalization=dict(mean=bank.manifest['train_mean'], std=bank.manifest['train_std']),
            split_sha256=bank.manifest['split_sha256']))
    print(json.dumps(dict(parameters=count, training_images=len(ids), epochs=config['epochs'], all_data=args.all_data)), flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first, config['epochs']+1):
        warmup=config['warmup_epochs']
        if epoch <= warmup:
            lr=config['learning_rate']*epoch/max(1,warmup)
        else:
            lr=config['learning_rate']*.5*(1+math.cos(math.pi*(epoch-1-warmup)/max(1,config['epochs']-warmup)))
        for group in optimizer.param_groups: group['lr']=lr
        model.train()
        epoch_shared.fill_(epoch)
        total_loss=torch.zeros((),device=device)
        weighted_correct=torch.zeros((),device=device)
        examples=0
        synchronize(device); start=time.perf_counter()
        for batch_index,(cpu_images,cpu_labels) in enumerate(loader):
            if args.smoke_batches and batch_index>=args.smoke_batches:break
            x=cpu_images.to(device,non_blocking=True,memory_format=torch.channels_last)
            labels=cpu_labels.to(device,non_blocking=True)
            x=degradation_augment(x,bank.mean,bank.std,aug_rng)
            x=cutout(x,config['cutout_size'],aug_rng)
            def closure():
                with autocast(device):
                    logits=model(x)
                    return F.cross_entropy(logits,labels),logits
            optimizer.zero_grad(set_to_none=True)
            loss,logits=closure()
            loss.backward();optimizer.step()
            steps += 1
            decay=min(config['ema_decay'],(1+steps)/(10+steps))
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-decay)
                # BN buffers copied, not averaged twice through two running means.
                for target, source in zip(ema.buffers(),model.buffers()): target.copy_(source)
            total_loss += loss.detach()*len(labels)
            examples += len(labels)
            pred=logits.detach().argmax(1)
            weighted_correct += (pred==labels).sum()
        synchronize(device); elapsed=time.perf_counter()-start
        assert torch.isfinite(total_loss), 'Non-finite training loss'
        row=dict(epoch=epoch,learning_rate=lr,train_loss=total_loss.item()/examples,
                 train_accuracy=100*weighted_correct.item()/examples,training_seconds=elapsed)
        if not args.all_data:
            for variant, network in [('raw',model),('ema',ema)]:
                metric=evaluate(network,bank,bank.val_indices,config['eval_batch_size'])
                row[variant+'_val_accuracy']=metric['accuracy']; row[variant+'_val_loss']=metric['loss']
                candidate=(metric['accuracy'],-metric['loss'])
                if candidate > (best[0],-best[1]):
                    best=(metric['accuracy'],metric['loss'])
                    save_model(args.output/'best.pt',network,epoch,metric,variant)
                    atomic_json(args.output/'best.json',dict(epoch=epoch,variant=variant,**metric))
        if args.all_data and epoch==config.get('selected_epoch'):
            save_model(args.output/'selected.pt',ema if config['final_variant']=='ema' else model,epoch,None,config['final_variant'])
        history.append(row)
        atomic_json(args.output/'history.json',history)
        with (args.output/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        atomic_json(args.output/'status.json',dict(epoch=epoch,total_epochs=config['epochs'],best_validation_accuracy=None if args.all_data else best[0],latest=row))
        # A recoverable checkpoint every 5 epochs, plus the first and final epochs.
        if epoch % 5 == 0 or epoch in (1,config['epochs']):
            atomic_torch_save(args.output/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),epoch=epoch,config=config,history=history,best=best,steps=steps,
                source_sha256=sources,split_sha256=bank.manifest['split_sha256'],order_rng=order_rng.get_state(),
                aug_rng=aug_rng.get_state(),mix_rng=mix_rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
        print(json.dumps(row),flush=True)
    if args.all_data:
        variant=config['final_variant']
        assert variant in ('raw','ema')
        save_model(args.output/'final.pt',ema if variant=='ema' else model,config['epochs'],None,variant)
    summary=dict(parameters=count,completed_epochs=len(history),training_images=len(ids),all_data=args.all_data,
                 best_validation=None if args.all_data else json.loads((args.output/'best.json').read_text()),
                 training_seconds=sum(r['training_seconds'] for r in history),
                 peak_allocated_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,test_accuracy=None,parent_checkpoint=provenance)
    atomic_json(args.output/'summary.json',summary)
    print('COMPLETE '+json.dumps(summary),flush=True)


if __name__=='__main__': main()
