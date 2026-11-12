"""Independent per-image RandAugment with deterministic epoch/item seeds.

Only the supplied cached training images are read. Worker RNG is reset per item,
so checkpoint resumption does not depend on hidden persistent-worker RNG state.
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,Sampler
from torchvision import transforms as T
from common import ROOT


class WideTrainingData(Dataset):
    def __init__(self,indices,manifest,config,epoch):
        self.indices=np.asarray(indices,dtype=np.int64)
        self.manifest=manifest;self.config=config;self.epoch=epoch
        self.images=None;self.labels=None
        fill=[round(255*x) for x in manifest['train_mean']]
        self.transform=T.Compose([
            T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=config['randaugment_ops'],magnitude=config['randaugment_magnitude'],
                          interpolation=T.InterpolationMode.BILINEAR,fill=fill),
            T.ToTensor(),T.Normalize(manifest['train_mean'],manifest['train_std'])])

    def __len__(self):return len(self.indices)

    def __getstate__(self):
        state=self.__dict__.copy();state['images']=None;state['labels']=None
        return state

    def __getitem__(self,item):
        if self.images is None:
            self.images=np.load(ROOT/'cache/train_images.npy',mmap_mode='r')
            self.labels=np.load(ROOT/'cache/labels.npy',mmap_mode='r')
        idx=int(self.indices[item])
        image=Image.fromarray(np.ascontiguousarray(self.images[idx].transpose(1,2,0)))
        seed=self.config['seed']+int(self.epoch.item())*1_000_003+idx
        rng=torch.Generator().manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(rng.get_state())
            image=self.transform(image)
        return image,int(self.labels[idx])


class EpochSampler(Sampler):
    def __init__(self,dataset,seed):self.dataset=dataset;self.seed=seed
    def __len__(self):return len(self.dataset)
    def __iter__(self):
        rng=torch.Generator().manual_seed(self.seed+int(self.dataset.epoch.item())*10_007)
        return iter(torch.randperm(len(self.dataset),generator=rng).tolist())
