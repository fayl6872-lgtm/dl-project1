"""Shared cached data, augmentation, evaluation, and checkpoint helpers."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
# Dataset files (train_images/, test_images/, *.csv) live in <repo>/data, shared by both pipelines.
DATA_DIR=ROOT.parent/'data'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temporary.replace(path)


def atomic_torch_save(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(data,temporary)
    temporary.replace(path)


def load_manifest():
    cache=ROOT/'cache'
    manifest=json.loads((cache/'data_manifest.json').read_text())
    for name,expected in manifest['source_csv_sha256'].items():
        assert hashlib.sha256((DATA_DIR/name).read_bytes()).hexdigest()==expected, f'{name} has changed'
    for split,expected in manifest['split_sha256'].items():
        assert hashlib.sha256((ROOT/'splits'/f'{split}.csv').read_bytes()).hexdigest()==expected
    return manifest


class ImageBank:
    """Keep the small 32x32 corpus as uint8 tensors on the selected device.

    This avoids repeated PNG decoding and Windows worker-process overhead.
    Individual batches are converted to float only when needed.
    """
    def __init__(self,device,test=False):
        self.device=torch.device(device)
        self.manifest=load_manifest()
        cache=ROOT/'cache'
        name='test_images.npy' if test else 'train_images.npy'
        self.images=torch.from_numpy(np.load(cache/name)).to(self.device)
        self.mean=torch.tensor(self.manifest['train_mean'],device=self.device,dtype=torch.float32).view(1,3,1,1)
        self.std=torch.tensor(self.manifest['train_std'],device=self.device,dtype=torch.float32).view(1,3,1,1)
        if not test:
            self.labels=torch.from_numpy(np.load(cache/'labels.npy')).to(self.device)
            self.train_indices=torch.from_numpy(np.load(cache/'train_indices.npy')).to(self.device)
            self.val_indices=torch.from_numpy(np.load(cache/'validation_indices.npy')).to(self.device)

    def batch(self,indices,augment=False,generator=None):
        x=self.images[indices].float().div_(255)
        if augment:
            # RandomCrop(32, padding=4), zero padding before normalization.
            n=len(indices)
            padded=F.pad(x,(4,4,4,4),value=0.)
            oy=torch.randint(0,9,(n,1,1,1),device=self.device,generator=generator)
            ox=torch.randint(0,9,(n,1,1,1),device=self.device,generator=generator)
            b=torch.arange(n,device=self.device)[:,None,None,None]
            c=torch.arange(3,device=self.device)[None,:,None,None]
            y=torch.arange(32,device=self.device)[None,None,:,None]+oy
            z=torch.arange(32,device=self.device)[None,None,None,:]+ox
            x=padded[b,c,y,z]
            flip=torch.rand((n,1,1,1),device=self.device,generator=generator)<.5
            x=torch.where(flip,x.flip(-1),x)
        return ((x-self.mean)/self.std).contiguous(memory_format=torch.channels_last)


def autocast(device):
    return torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda')


def synchronize(device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model,bank,indices,batch_size=256,return_predictions=False):
    model.eval()
    total_loss=torch.zeros((),device=bank.device)
    correct=torch.zeros((),dtype=torch.int64,device=bank.device)
    predictions=[]
    for start in range(0,len(indices),batch_size):
        idx=indices[start:start+batch_size]
        with autocast(bank.device):
            logits=model(bank.batch(idx))
            loss=F.cross_entropy(logits,bank.labels[idx],reduction='sum')
        pred=logits.argmax(1)
        total_loss+=loss
        correct+=(pred==bank.labels[idx]).sum()
        if return_predictions:
            predictions.extend(pred.cpu().tolist())
    result=dict(loss=total_loss.item()/len(indices),accuracy=100*correct.item()/len(indices),
                correct=int(correct.item()),total=len(indices))
    if return_predictions:result['predictions']=predictions
    return result
