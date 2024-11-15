import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net
def compute_params_score(model, resolution, batch_size,image_channel,is_3D,gpu=0):
    model.train()
    model.requires_grad_(True)

    model.zero_grad()

    if gpu:
        # torch.cuda.set_device(gpu)
        model = model.cuda()
    else:
        model = model.to('cpu')

    network_weight_gaussian_init(model)
    if is_3D:
        input = torch.randn(size=[batch_size, image_channel, resolution, resolution, resolution])
    else:
        input = torch.randn(size=[batch_size, image_channel, resolution, resolution])
    if gpu:
        input = input.cuda()
    else:
        input = input.to('cpu')

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params