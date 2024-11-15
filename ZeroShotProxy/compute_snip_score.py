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
import types
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_layer_metric_array(net, metric, mode): 
    metric_array = []

    for layer in net.modules():
        if mode=='channel' and hasattr(layer,'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv3d):
            metric_array.append(metric(layer))
    
    return metric_array



def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
def snip_forward_conv3d(self, x):
        return F.conv3d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)
def compute_snip_per_weight(net, inputs, batch_size, mode , gpu):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv3d):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)
        if isinstance(layer, nn.Conv3d):
            layer.forward = types.MethodType(snip_forward_conv3d, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    _, outputs = net.forward(inputs)
    num_classes = outputs.shape[1]
    y = torch.randint(low=0, high=num_classes, size=[batch_size]).cuda()
    one_hot_y = F.one_hot(y, num_classes).float()
    if gpu:
        one_hot_y = one_hot_y.cuda()
    else:
        one_hot_y = one_hot_y.cpu()
    loss = F.cross_entropy(outputs, one_hot_y)
    loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)
    
    grads_abs = get_layer_metric_array(net, snip, mode)

    return grads_abs

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
def compute_snip_score(model, resolution, batch_size,image_channel,is_3D,gpu=0):
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
    grads_abs_list = compute_snip_per_weight(net=model, inputs=input,batch_size=batch_size, mode='', gpu = gpu)
    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('!!!')
    return score
