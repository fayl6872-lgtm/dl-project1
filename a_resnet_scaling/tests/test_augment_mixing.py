import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from augment_mixing import cutout, mix_batch, mixed_loss
from resnet import build_model
from common import ROOT


class AdvancedTests(unittest.TestCase):
    def test_cutmix_area_matches_label_weight(self):
        x=torch.arange(8,dtype=torch.float32)[:,None,None,None].expand(8,3,32,32).clone()
        labels=torch.arange(8)
        config=dict(mix_probability=1.,cutmix_fraction=1.,cutmix_alpha=1.)
        for seed in range(15):
            mixed,a,b,lam=mix_batch(x,labels,config,np.random.default_rng(seed),torch.Generator().manual_seed(seed))
            expected=lam*a.float()+(1-lam)*b.float()
            torch.testing.assert_close(mixed.mean((1,2,3)),expected)
            self.assertGreaterEqual(lam,0); self.assertLessEqual(lam,1)
        torch.testing.assert_close(x[:,0,0,0],labels.float())

    def test_mixup_loss_matches_soft_targets(self):
        config=dict(mix_probability=1.,cutmix_fraction=0.,mixup_alpha=.2)
        labels=torch.arange(8)%3; x=torch.randn(8,3,32,32)
        _,a,b,lam=mix_batch(x,labels,config,np.random.default_rng(7),torch.Generator().manual_seed(7))
        logits=torch.randn(8,3,requires_grad=True)
        target=lam*torch.nn.functional.one_hot(a,3)+(1-lam)*torch.nn.functional.one_hot(b,3)
        torch.testing.assert_close(mixed_loss(logits,a,b,lam),torch.nn.functional.cross_entropy(logits,target.float()))

    def test_cutout_and_budget(self):
        x=torch.ones(8,3,32,32)
        result=cutout(x,16,torch.Generator().manual_seed(5))
        self.assertTrue((result==0).any()); self.assertTrue((result==1).any())
        self.assertTrue((x==1).all())
        config=json.loads((ROOT/'configs/a1_dev_cutout.json').read_text())
        model=build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()),4368690)


if __name__=='__main__': unittest.main()
