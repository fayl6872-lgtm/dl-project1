"""Training-only stochastic image-quality variation, never applied to evaluation."""
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur


def degradation_augment(x,mean,std,generator):
    pixels=(x*std+mean).clamp(0,1)
    # One shared operation type per batch; independent per-image strength/mask.
    op=int(torch.randint(0,4,(),generator=generator,device=x.device).item())
    mask=torch.rand((len(x),1,1,1),generator=generator,device=x.device)<.35
    if op==0:
        strength=torch.rand((len(x),1,1,1),generator=generator,device=x.device)*.04
        altered=(pixels+torch.randn(pixels.shape,generator=generator,device=x.device)*strength).clamp(0,1)
    elif op==1:
        size=int(torch.randint(22,32,(),generator=generator,device=x.device).item())
        altered=F.interpolate(F.interpolate(pixels,size=(size,size),mode='bilinear',align_corners=False,antialias=True),size=(32,32),mode='bilinear',align_corners=False,antialias=True)
    elif op==2:
        altered=gaussian_blur(pixels,[3,3],[.6,.6])
    else:
        strength=.65+torch.rand((len(x),1,1,1),generator=generator,device=x.device)*.7
        altered=(pixels*strength).clamp(0,1)
    return ((torch.where(mask,altered,pixels)-mean)/std).contiguous(memory_format=torch.channels_last)
