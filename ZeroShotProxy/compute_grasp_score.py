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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def get_layer_metric_array(net, metric, mode): 
    metric_array = []

    for layer in net.modules():
        if mode=='channel' and hasattr(layer,'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    
    return metric_array

def compute_grasp_per_weight(net, inputs,batch_size, mode, gpu):

    # get all applicable weights
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv3d):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True) # TODO isn't this already true?

    # NOTE original code had some input/target splitting into 2
    # I am guessing this was because of GPU mem limit
    net.zero_grad()
    #forward/grad pass #1
    grad_w = None
    #TODO get new data, otherwise num_iters is useless!
    _, outputs = net.forward(inputs)
    num_classes = outputs.shape[1]
    y = torch.randint(low=0, high=num_classes, size=[batch_size]).cuda()
    one_hot_y = F.one_hot(y, num_classes).float()
    if gpu:
        one_hot_y = one_hot_y.cuda()
    else:
        one_hot_y = one_hot_y.cpu()
    loss = F.cross_entropy(outputs, one_hot_y)
    grad_w_p = autograd.grad(loss, weights, allow_unused=True)
    if grad_w is None:
        grad_w = list(grad_w_p)
    else:
        for idx in range(len(grad_w)):
            grad_w[idx] += grad_w_p[idx]

    # forward/grad pass #2
    _, outputs = net.forward(inputs)
    num_classes = outputs.shape[1]
    y = torch.randint(low=0, high=num_classes, size=[batch_size]).cuda()
    one_hot_y = F.one_hot(y, num_classes).float()
    if gpu:
        one_hot_y = one_hot_y.cuda()
    else:
        one_hot_y = one_hot_y.cpu()
    loss = F.cross_entropy(outputs, one_hot_y)
    grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)
    
    # accumulate gradients computed in previous step and call backwards
    z, count = 0,0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv3d):
            if grad_w[count] is not None:
                z += (grad_w[count].data * grad_f[count]).sum()
            count += 1
    z.backward()

    # compute final sensitivity metric and put in grads
    def grasp(layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad   # -theta_q Hg
            #NOTE in the grasp code they take the *bottom* (1-p)% of values
            #but we take the *top* (1-p)%, therefore we remove the -ve sign
            #EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)
    
    grads = get_layer_metric_array(net, grasp, mode)

    return grads
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
def compute_grasp_score(model, resolution, batch_size,image_channel,is_3D,gpu=0):
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
    grads_abs_list = compute_grasp_per_weight(net=model, inputs=input,batch_size=batch_size, mode='', gpu = gpu)
    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('!!!')
    return score
