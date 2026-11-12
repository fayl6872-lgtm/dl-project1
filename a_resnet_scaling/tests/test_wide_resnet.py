import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from wide_resnet import WideBlock,build_model
from randaugment_data import WideTrainingData,EpochSampler
from common import ROOT


class WideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_identity_including_negative_inputs(self):
        m=WideBlock(8,8).eval()
        torch.nn.init.zeros_(m.conv2.weight)
        x=torch.randn(2,8,16,16,requires_grad=True)
        y=m(x);torch.testing.assert_close(y,x)
        y.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x))

    def test_budget_resolution_and_gradient(self):
        c=json.loads((ROOT/'configs/a2_dev_wrn.json').read_text());m=build_model(c)
        self.assertEqual(sum(p.numel() for p in m.parameters()),4480474)
        shapes=[];handle=m.stages.register_forward_hook(lambda module,inputs,output:shapes.append(output.shape))
        y=m(torch.randn(2,3,32,32));handle.remove()
        self.assertEqual(tuple(shapes[0]),(2,224,8,8));self.assertEqual(tuple(y.shape),(2,10))
        torch.nn.functional.cross_entropy(y,torch.tensor([0,9])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        with self.assertRaises(AssertionError):build_model({**c,'channels':[64,128,256]})

    @unittest.skipUnless((ROOT/'cache/data_manifest.json').exists(),'needs prepare_data.py output')
    def test_augmentation_and_epoch_reproducibility(self):
        c=json.loads((ROOT/'configs/a2_dev_wrn.json').read_text())
        manifest=json.loads((ROOT/'cache/data_manifest.json').read_text())
        ids=np.load(ROOT/'cache/train_indices.npy')[:12]
        epoch=torch.tensor(1)
        data=WideTrainingData(ids,manifest,c,epoch)
        state=torch.get_rng_state().clone();a,y=data[0];b,z=data[0]
        torch.testing.assert_close(a,b);self.assertEqual(y,z)
        torch.testing.assert_close(state,torch.get_rng_state())
        sampler=EpochSampler(data,c['seed']);order=list(sampler)
        self.assertEqual(order,list(sampler));self.assertEqual(sorted(order),list(range(12)))
        epoch.fill_(2);other,_=data[0]
        self.assertFalse(torch.equal(a,other));self.assertNotEqual(order,list(sampler))
        self.assertTrue(torch.isfinite(other).all())


if __name__=='__main__':unittest.main()
