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

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import types
# from ..p_utils import get_layer_metric_array, reshape_elements


def get_layer_metric_array(net, metric, mode): 
    metric_array = []

    for layer in net.modules():
        if mode=='channel' and hasattr(layer,'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    
    return metric_array


def reshape_elements(elements, shapes):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e,sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).cuda())
        return ret_grads
    if type(elements[0]) == list:
        outer = []
        for e,sh in zip(elements, shapes):
            outer.append(broadcast_val(e,sh))
        return outer
    else:
        return broadcast_val(elements, shapes)


def fisher_forward_conv2d(self, x):
    x = F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
    #intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act

def fisher_forward_conv3d(self, x):
    x = F.conv3d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
    #intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act

def fisher_forward_linear(self, x):
    x = F.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act

def compute_fisher_per_weight(net, batch_size, inputs, gpu,  mode='', split_data=1):
    net.train()
    net.requires_grad_(True)
    net.zero_grad()
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv3d):
            #variables/op needed for fisher computation
            layer.fisher = None
            layer.act = 0.
            layer.dummy = nn.Identity()

            #replace forward method of conv/linear
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(fisher_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(fisher_forward_linear, layer)
            if isinstance(layer, nn.Conv3d):
                layer.forward = types.MethodType(fisher_forward_conv3d, layer)

            #function to call during backward pass (hooked on identity op at output of layer)
            def hook_factory(layer):
                def hook(module, grad_input, grad_output):
                    act = layer.act.detach()
                    grad = grad_output[0].detach()
                    if len(act.shape) > 2:
                        g_nk = torch.sum((act * grad), list(range(2,len(act.shape))))
                    else:
                        g_nk = act * grad
                    del_k = g_nk.pow(2).mean(0).mul(0.5)
                    if layer.fisher is None:
                        layer.fisher = del_k
                    else:
                        layer.fisher += del_k
                    del layer.act #without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
                return hook

            #register backward hook on identity fcn to compute fisher info
            layer.dummy.register_backward_hook(hook_factory(layer))
    # N = inputs.shape[0]
    # for sp in range(split_data):
    #     st=sp*N//split_data
    #     en=(sp+1)*N//split_data


    _, outputs = net(inputs)

    num_classes = outputs.shape[1]
    y = torch.randint(low=0, high=num_classes, size=[batch_size]).cuda()
    one_hot_y = F.one_hot(y, num_classes).float()
    if gpu:
        outputs = outputs.cuda()
        one_hot_y = one_hot_y.cuda()
    else:
        outputs = outputs.to('cpu')
        one_hot_y = one_hot_y.to('cpu')
    loss = F.cross_entropy(outputs, one_hot_y)
    loss.backward()

    # retrieve fisher info
    def fisher(layer):
        if layer.fisher is not None:
            return torch.abs(layer.fisher.detach())
        else:
            return torch.zeros(layer.weight.shape[0]) #size=ch

    grads_abs_ch = get_layer_metric_array(net, fisher, mode)

    #broadcast channel value here to all parameters in that channel
    #to be compatible with stuff downstream (which expects per-parameter metrics)
    #TODO cleanup on the selectors/apply_prune_mask side (?)
    shapes = get_layer_metric_array(net, lambda l : l.weight.shape[1:], mode)

    grads_abs = reshape_elements(grads_abs_ch, shapes)
    # grads_score = [0] * len(grads_abs)
    # for i in range(len(grads_abs)):
    #     grads_score[i] = torch.mean(torch.sum(grads_abs[i], dim=0))
    # cpu_tensors = [t.cpu() for t in grads_score]
    # stacked_tensors = torch.stack(cpu_tensors)
    # result_tensor = torch.sum(stacked_tensors)
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
def compute_fisher_score(model, resolution, batch_size,image_channel,is_3D,gpu=0):
    model.train()
    model.requires_grad_(True)

    model.zero_grad()

    if gpu:
        # torch.cuda.set_device(0)
        model = model.cuda()
    else:
        model = model.to('cpu')

    model = network_weight_gaussian_init(model)
    if is_3D:
        input = torch.randn(size=[batch_size, image_channel, resolution, resolution, resolution])
    else:
        input = torch.randn(size=[batch_size, image_channel, resolution, resolution])
    if gpu:
        input = input.cuda()
    else:
        input = input.to('cpu')
    grads_abs_list = compute_fisher_per_weight(net=model,batch_size=batch_size, inputs=input, mode='',gpu=gpu)
    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('!!!')


    return score
