'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

This file is modified from:
https://github.com/VITA-Group/TENAS
'''

import argparse
import os, sys
import time

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
def get_ntk_n(networks, image_channel,is_3D,recalbn=0, train_mode=False, num_batch=None,
              batch_size=None, image_size=None, gpu=None):
    if gpu:
        device = torch.device('cuda:{}'.format(0))
    else:
        device = torch.device('cpu')
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]
    for i in range(num_batch):
        if is_3D:
            inputs = torch.randn((batch_size, image_channel, image_size, image_size, image_size), device=device)
        else:
            inputs = torch.randn((batch_size, image_channel, image_size, image_size), device=device)
        # inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            if gpu:
                network = network.cuda()
                inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            else:
                network = network.cpu()
                inputs_ = inputs.clone().cpu()

            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                if gpu:
                    torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.linalg.eigh(ntk, UPLO='L')   # ascending
        # conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True))
    return conds



def compute_NTK_score(model, resolution, batch_size, image_channel, is_3D, gpu=0):
    ntk_score = get_ntk_n([model], image_channel,is_3D,recalbn=0, train_mode=True, num_batch=1,
                           batch_size=batch_size, image_size=resolution, gpu=gpu,)[0]
    return -1 * ntk_score
