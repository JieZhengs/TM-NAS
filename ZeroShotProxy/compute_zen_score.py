'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import time


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


def compute_ZENSCORE_score(model, resolution, batch_size, image_channel, is_3D, repeat, gpu, mixup_gamma=1e-2, fp16=False):
    info = {}
    nas_score_list = []
    if gpu:
        print('使用GPU！')
        device = torch.device('cuda:{}'.format(0))
        model = model.to(device)
    else:
        print('使用CPU！')
        device = torch.device('cpu')
        model = model.to(device)
    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)
            if is_3D:
                input = torch.randn(size=[batch_size, image_channel, resolution, resolution, resolution], device=device, dtype=dtype)
                input2 = torch.randn(size=[batch_size, image_channel, resolution, resolution, resolution], device=device, dtype=dtype)
            else:
                input = torch.randn(size=[batch_size, image_channel, resolution, resolution], device=device, dtype=dtype)
                input2 = torch.randn(size=[batch_size, image_channel, resolution, resolution], device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2
            output, _ = model.forward(input)
            mixup_output,_  = model.forward(mixup_input)
            nas_score = torch.sum(torch.abs(output - mixup_output))
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)

    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)
    return info
