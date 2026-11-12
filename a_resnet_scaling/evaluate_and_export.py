"""Evaluate one checkpoint under test-time-augmentation policies, or export a submission CSV.

Used for every model in this folder (A0-A3 and the robustness probe).

  Validation mode:  python evaluate_and_export.py --checkpoint runs/a2_dev_wrn/best.pt
  Export mode:      python evaluate_and_export.py --checkpoint runs/a3_full_sam/final.pt \
                        --policy flip --export submissions/a3_sam.csv

Test-time views reuse one model: the parameter count is unchanged. No averaging
of predictions from multiple checkpoints or test-label lookup is performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from wide_resnet import build_model
from common import DATA_DIR, ROOT, ImageBank, atomic_json, autocast


POLICIES=['none','flip','shift_flip','scale_flip','crop36']


def views(x, policy):
    yield x
    if policy in ('flip','shift_flip','scale_flip','crop36'):
        yield x.flip(-1)
    if policy in ('scale_flip','crop36'):
        enlarged=F.interpolate(x,size=(36,36),mode='bilinear',align_corners=False,antialias=True)
        positions=[(2,2)] if policy=='scale_flip' else [(0,0),(0,4),(4,0),(4,4),(2,2)]
        for y,z in positions:
            crop=enlarged[:,:,y:y+32,z:z+32]
            yield crop
            yield crop.flip(-1)
        if policy=='scale_flip':
            reduced=F.interpolate(x,size=(28,28),mode='bilinear',align_corners=False,antialias=True)
            small=F.pad(reduced,(2,2,2,2),mode='reflect')
            yield small
            yield small.flip(-1)
    if policy == 'shift_flip':
        padded=F.pad(x,(2,2,2,2),mode='reflect')
        for y,z in [(0,0),(0,4),(4,0),(4,4)]:
            crop=padded[:,:,y:y+32,z:z+32]
            yield crop
            yield crop.flip(-1)


@torch.inference_mode()
def probabilities(model, bank, indices, policy):
    chunks=[]
    for start in range(0,len(indices),256):
        x=bank.batch(indices[start:start+256])
        total=None; count=0
        for view in views(x,policy):
            with autocast(bank.device): logits=model(view)
            p=logits.float().softmax(1)
            total=p if total is None else total+p
            count+=1
        chunks.append((total/count).cpu().numpy())
    return np.concatenate(chunks)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--export',type=Path)
    parser.add_argument('--policy',choices=POLICIES)
    args=parser.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=True
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=True)
    assert not checkpoint['config']['smoke_batches']
    model=build_model(checkpoint['config']).to(device=device,memory_format=torch.channels_last).eval()
    model.load_state_dict(checkpoint['model'])
    parameters=sum(p.numel() for p in model.parameters())
    assert parameters==checkpoint['parameters'] and parameters<=5_000_000
    bank=ImageBank(device,test=args.export is not None)
    assert checkpoint['normalization']==dict(mean=bank.manifest['train_mean'],std=bank.manifest['train_std'])
    if args.export:
        assert args.policy, 'Choose inference policy on held-out validation BEFORE test export.'
        ids=torch.arange(len(bank.images),device=device)
        probs=probabilities(model,bank,ids,args.policy)
        assert np.isfinite(probs).all() and (probs>=0).all()
        assert np.allclose(probs.sum(1),1.,atol=1e-5)
        rows=list(csv.DictReader((DATA_DIR/'test.csv').open()))
        sample=list(csv.DictReader((DATA_DIR/'sample_submission.csv').open()))
        mapping={r['id']:int(p) for r,p in zip(rows,probs.argmax(1))}
        assert len(rows)==len(probs)==len(sample)==len(mapping)==2000
        assert set(mapping)=={r['id'] for r in sample}
        args.export.parent.mkdir(parents=True,exist_ok=True)
        with args.export.open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['id','label'])
            writer.writerows((r['id'],mapping[r['id']]) for r in sample)
        saved=list(csv.DictReader(args.export.open()))
        assert [r['id'] for r in saved]==[r['id'] for r in sample]
        assert all(0<=int(r['label'])<=9 for r in saved)
        result=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    checkpoint_epoch=checkpoint['epoch'],variant=checkpoint.get('variant','raw'),parameters=parameters,
                    parent_checkpoint=checkpoint.get('parent_checkpoint'),
                    cumulative_training_epochs=checkpoint['epoch']+checkpoint.get('parent_checkpoint',{}).get('cumulative_epochs',checkpoint.get('parent_checkpoint',{}).get('epoch',0)),
                    policy=args.policy,rows=2000,trained_on_all_data=checkpoint['config'].get('all_data',False),
                    validation_of_this_checkpoint=checkpoint['validation'],
                    csv_sha256=hashlib.sha256(args.export.read_bytes()).hexdigest())
        comparison_path=args.checkpoint.parent/'inference_comparison.json'
        result['validation_with_selected_inference']=None
        if comparison_path.exists() and not result['trained_on_all_data']:
            comparison=json.loads(comparison_path.read_text())
            assert Path(comparison['checkpoint']).resolve()==args.checkpoint.resolve()
            result['validation_with_selected_inference']=comparison['policies'][args.policy]
        if result['trained_on_all_data']:
            result['validation_note']='Recipe score belongs to the held-out comparison model, not this all-data checkpoint.'
        atomic_json(args.export.with_suffix('.json'),result)
        np.save(args.export.with_suffix('.probabilities.npy'),probs)
        print(json.dumps(result,indent=2))
    else:
        assert not checkpoint['config'].get('all_data',False), 'Cannot validate on data used in all-data training.'
        assert checkpoint['split_sha256']==bank.manifest['split_sha256']
        truth=bank.labels[bank.val_indices].cpu().numpy()
        result=dict(checkpoint=str(args.checkpoint.resolve()),epoch=checkpoint['epoch'],variant=checkpoint.get('variant','raw'),
                    parameters=parameters,validation_examples=len(truth),policies={})
        for policy in ([args.policy] if args.policy else POLICIES):
            probs=probabilities(model,bank,bank.val_indices,policy)
            pred=probs.argmax(1)
            loss=float(-np.log(np.clip(probs[np.arange(len(truth)),truth],1e-12,1)).mean())
            metric=dict(accuracy=float(100*np.mean(pred==truth)),correct=int((pred==truth).sum()),loss=loss)
            result['policies'][policy]=metric
            np.save(args.checkpoint.parent/f'validation_probabilities_{policy}.npy',probs)
            print(policy,json.dumps(metric),flush=True)
        result['selected_policy']=max(result['policies'],key=lambda p:(result['policies'][p]['accuracy'],-result['policies'][p]['loss']))
        atomic_json(args.checkpoint.parent/'inference_comparison.json',result)


if __name__=='__main__': main()
