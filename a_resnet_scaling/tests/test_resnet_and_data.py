"""Meaningful smoke checks: residual identity, budget, gradients, split isolation."""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from resnet import ResidualBlock, build_model
from common import ROOT,ImageBank


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_residual_identity_and_gradient(self):
        block=ResidualBlock(8,8).eval()
        for parameter in block.branch.parameters():
            torch.nn.init.zeros_(parameter)
        x=torch.ones(2,8,8,8,requires_grad=True)
        y=block(x)
        torch.testing.assert_close(y,x)
        y.sum().backward()
        torch.testing.assert_close(x.grad,torch.ones_like(x))

    def test_model_budget_shapes_and_backward(self):
        config=json.loads((ROOT/'configs/a0_resnet18_baseline.json').read_text())
        model=build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()),2797610)
        logits=model(torch.randn(2,3,32,32))
        self.assertEqual(tuple(logits.shape),(2,10))
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,9])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        bad={**config,'base_width':64}
        with self.assertRaises(AssertionError):build_model(bad)

    @unittest.skipUnless((ROOT/'cache/data_manifest.json').exists(),'needs prepare_data.py output')
    def test_split_isolation_and_evaluation_preprocessing(self):
        bank=ImageBank('cpu')
        train=set(bank.train_indices.tolist());val=set(bank.val_indices.tolist())
        self.assertFalse(train&val)
        self.assertEqual(len(train),45000);self.assertEqual(len(val),5000)
        self.assertEqual(len(train|val),50000)
        self.assertEqual(torch.bincount(bank.labels[bank.val_indices]).tolist(),[500]*10)
        ids=bank.val_indices[:8]
        x=bank.batch(ids)
        expected=(bank.images[ids].float()/255-bank.mean)/bank.std
        torch.testing.assert_close(x,expected)
        torch.testing.assert_close(x,bank.batch(ids))
        a=bank.batch(ids,augment=True,generator=torch.Generator().manual_seed(123))
        b=bank.batch(ids,augment=True,generator=torch.Generator().manual_seed(123))
        torch.testing.assert_close(a,b)
        self.assertEqual(tuple(a.shape),(8,3,32,32))
        self.assertTrue(torch.isfinite(a).all())


if __name__=='__main__':unittest.main(verbosity=2)
