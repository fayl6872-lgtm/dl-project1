"""Two-pass SAM with exact weight restoration and one BN update per minibatch.
Reference: Foret et al., https://arxiv.org/abs/2010.01412 .
"""
import torch


def sam_update(model, optimizer, closure, rho):
    optimizer.zero_grad(set_to_none=True)
    loss, logits = closure()
    loss.backward()
    parameters=[p for p in model.parameters() if p.grad is not None]
    norm=torch.linalg.vector_norm(torch.stack([p.grad.float().norm(2) for p in parameters]),2)
    if not torch.isfinite(norm):raise FloatingPointError('Non-finite SAM gradient')
    originals=[p.detach().clone() for p in parameters]
    buffers=[(b,b.detach().clone()) for b in model.buffers()]
    try:
        with torch.no_grad():
            scale=rho/(norm+1e-12)
            for p in parameters:p.add_(p.grad*scale)
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss,_=closure()
        if not torch.isfinite(perturbed_loss):raise FloatingPointError('Non-finite perturbed SAM loss')
        perturbed_loss.backward()
    finally:
        with torch.no_grad():
            for p,old in zip(parameters,originals):p.copy_(old)
            for b,old in buffers:b.copy_(old)
    optimizer.step()
    return loss.detach(),logits.detach(),perturbed_loss.detach()
