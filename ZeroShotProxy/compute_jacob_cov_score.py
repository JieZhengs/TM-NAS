# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_batch_jacobian(net, x):
    x.requires_grad_(True)
    y,_ = net(x)
    y.backward(torch.ones_like(y))

    jacob = x.grad.detach()
    x.requires_grad_(False)
    return jacob

def eval_score(jacob):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))
def compute_jacob_cov(net, inputs):
    # Compute gradients (but don't apply them)
    net.zero_grad()

    jacobs = get_batch_jacobian(net, inputs)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    jc = eval_score(jacobs)


    return jc

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
def compute_jacob_cov_score(model, resolution, batch_size,image_channel,is_3D,gpu=0):
    model.train()
    model.requires_grad_(True)

    model.zero_grad()

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    network_weight_gaussian_init(model)
    if is_3D:
        input = torch.randn(size=[batch_size, image_channel, resolution, resolution, resolution])
    else:
        input = torch.randn(size=[batch_size, image_channel, resolution, resolution])
    if gpu is not None:
        input = input.cuda(gpu)

    score = compute_jacob_cov(net=model, inputs=input)
    return score