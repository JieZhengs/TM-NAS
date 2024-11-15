'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''



import os, sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
def getgrad(model:torch.nn.Module, grad_dict:dict, step_iter=0):
    if step_iter==0:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv3d) or isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                # print(mod.weight.grad.data.size())
                # print(mod.weight.data.size())
                grad_dict[name]=[mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv3d) or isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                grad_dict[name].append(mod.weight.grad.data.cpu().reshape( -1).numpy())
    return grad_dict

def caculate_zico(grad_dict):
    allgrad_array=None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx]))
    return nsr_mean_sum_abs

def getzico(network, inputs,batch_size, repeat,gpu):
    grad_dict= {}
    network.train()
    network.zero_grad()
    for i in range(repeat):
        # data = inputs
        _, logits = network(inputs)
        num_classes = logits.shape[1]
        y = torch.randint(low=0, high=num_classes, size=[batch_size]).cuda()
        one_hot_y = F.one_hot(y, num_classes).float()
        if gpu:
            one_hot_y = one_hot_y.cuda()
        else:
            one_hot_y = one_hot_y.cpu()
        loss = F.cross_entropy(logits, one_hot_y)
        loss.backward()
        grad_dict= getgrad(network, grad_dict, i)
    res = caculate_zico(grad_dict)
    return res


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
def compute_zico_score(model, resolution, batch_size,image_channel,is_3D,repeat=32, gpu=0):
    model.train()
    model.requires_grad_(True)

    model.zero_grad()

    if gpu:
        # torch.cuda.set_device(gpu)
        model = model.cuda()
    else:
        model = model.cpu()
    network_weight_gaussian_init(model)
    if is_3D:
        input = torch.randn(size=[batch_size, image_channel, resolution, resolution, resolution])
    else:
        input = torch.randn(size=[batch_size, image_channel, resolution, resolution])
    if gpu:
        input = input.cuda()
    else:
        input = input.cpu()

    score = getzico(network=model, inputs=input, batch_size=batch_size, repeat=repeat, gpu=gpu)
    return score