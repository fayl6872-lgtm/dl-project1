import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sam import sam_update


class SAMTests(unittest.TestCase):
    def test_quadratic_step_matches_hand_calculation(self):
        model=torch.nn.Linear(2,1,bias=False).double()
        with torch.no_grad():model.weight.copy_(torch.tensor([[1.,2.]],dtype=torch.double))
        original=model.weight.detach().clone();rho=.05;lr=.1
        optimizer=torch.optim.SGD(model.parameters(),lr=lr)
        def closure():
            value=model.weight
            return .5*value.square().sum(),value
        sam_update(model,optimizer,closure,rho)
        perturbation=rho*original/original.norm()
        expected=original-lr*(original+perturbation)
        torch.testing.assert_close(model.weight,expected,rtol=1e-7,atol=1e-9)

    def test_batchnorm_updates_once(self):
        torch.manual_seed(42)
        model=torch.nn.Sequential(torch.nn.BatchNorm1d(3),torch.nn.Linear(3,2))
        x=torch.randn(8,3);y=torch.arange(8)%2
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        def closure():
            logits=model(x)
            return torch.nn.functional.cross_entropy(logits,y),logits
        sam_update(model,optimizer,closure,.05)
        self.assertEqual(model[0].num_batches_tracked.item(),1)
        torch.testing.assert_close(model[0].running_mean,.1*x.mean(0))

    def test_exception_restores_weights_and_first_pass_buffers(self):
        model=torch.nn.Sequential(torch.nn.BatchNorm1d(3),torch.nn.Linear(3,2))
        x=torch.randn(8,3);y=torch.arange(8)%2
        original=[p.detach().clone() for p in model.parameters()];calls=0
        def closure():
            nonlocal calls
            calls+=1
            logits=model(x)
            if calls==2:raise ValueError('injected failure')
            return torch.nn.functional.cross_entropy(logits,y),logits
        with self.assertRaisesRegex(ValueError,'injected failure'):
            sam_update(model,torch.optim.SGD(model.parameters(),lr=.1),closure,.05)
        for p,old in zip(model.parameters(),original):torch.testing.assert_close(p,old,rtol=0,atol=0)
        self.assertEqual(model[0].num_batches_tracked.item(),1)


if __name__=='__main__':unittest.main()
