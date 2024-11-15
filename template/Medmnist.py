"""
# !/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import load_dataset.data_loader_Medmnist as data_loader
import os
import numpy as np
import math
import time
import copy
from datetime import datetime
import multiprocessing
from utils import Utils
from torchprofile import profile_macs
from template.drop import drop_path
from sklearn.metrics import cohen_kappa_score, roc_auc_score, f1_score, balanced_accuracy_score
from ZeroShotProxy.compute_zen_score import compute_ZENSCORE_score
from ZeroShotProxy.compute_fisher_score import compute_fisher_score
from ZeroShotProxy.compute_flops_score import compute_flops_score
from ZeroShotProxy.compute_params_score import compute_params_score
from ZeroShotProxy.compute_gradnorm_score import compute_GRADNORM_score
from ZeroShotProxy.compute_grasp_score import compute_grasp_score
from ZeroShotProxy.compute_NASWOT_score import compute_NASWOT_score
from ZeroShotProxy.compute_NTK_score import compute_NTK_score
from ZeroShotProxy.compute_snip_score import compute_snip_score
from ZeroShotProxy.compute_syncflow_score import compute_SYNCFLOW_score
from ZeroShotProxy.compute_zico_score import compute_zico_score
from ZeroShotProxy.compute_our_score import compute_our_score
from tqdm import tqdm
import sys
from template.kan import KAN, KANLinear
import warnings

# ignore warnings
warnings.filterwarnings("ignore")
class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()
class SELayer(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25):
        super(SELayer, self).__init__()
        reduce_chs = max(1, int(in_chs * se_ratio))
        self.act_fn = F.relu
        self.gate_fn = sigmoid
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x
class SELayer3D(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25):
        super(SELayer3D, self).__init__()
        reduce_chs = max(1, int(in_chs * se_ratio))
        self.act_fn = F.relu
        self.gate_fn = sigmoid
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv3d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv3d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x
class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, expansion=3, stride=1, dilation=1, act_func='h_swish',
                 attention=False, drop_connect_rate=0.0, dense=False, affine=True):
        super(MBConv, self).__init__()
        interChannels = expansion * C_out
        self.op1 = nn.Sequential(
            nn.Conv2d(C_in, interChannels, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(interChannels, affine=affine)
        )
        self.op2 = nn.Sequential(
            nn.Conv2d(interChannels, interChannels, kernel_size=kernel_size, stride=stride,
                      padding=int((kernel_size - 1) / 2) * dilation, bias=False, dilation=dilation,
                      groups=interChannels),
            nn.BatchNorm2d(interChannels, affine=affine)
        )
        self.op3 = nn.Sequential(
            nn.Conv2d(interChannels, C_out, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(C_out, affine=affine)
        )

        if act_func == 'relu':
            self.act_func = nn.ReLU(inplace=True)
        else:
            self.act_func = Hswish(inplace=True)
        if attention:
            self.se = SELayer(interChannels)
        else:
            self.se = nn.Sequential()
        self.drop_connect_rate = drop_connect_rate
        self.stride = stride
        self.dense = int(dense)
        self.C_in = C_in
        self.C_out = C_out

    def forward(self, x):
        out = self.op1(x)
        out = self.act_func(out)
        out = self.op2(out)
        out = self.act_func(out)
        out = self.se(out)
        out = self.op3(out)

        if self.drop_connect_rate > 0:
            out = drop_path(out, drop_prob=self.drop_connect_rate, training=self.training)
        if self.stride == 1 and self.dense:
            out = torch.cat([x, out], dim=1)
        elif self.stride == 1 and self.C_in == self.C_out:
            out = out + x
        return out
class MBConv3D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, expansion=3, stride=1, dilation=1, act_func='h_swish',
                 attention=False, drop_connect_rate=0.0, dense=False, affine=True):
        super(MBConv3D, self).__init__()
        interChannels = expansion * C_out
        self.op1 = nn.Sequential(
            nn.Conv3d(C_in, interChannels, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            nn.BatchNorm3d(interChannels, affine=affine)
        )
        self.op2 = nn.Sequential(
            nn.Conv3d(interChannels, interChannels, kernel_size=kernel_size, stride=stride,
                      padding=int((kernel_size - 1) / 2) * dilation, bias=False, dilation=dilation,
                      groups=interChannels),
            nn.BatchNorm3d(interChannels, affine=affine)
        )
        self.op3 = nn.Sequential(
            nn.Conv3d(interChannels, C_out, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            nn.BatchNorm3d(C_out, affine=affine)
        )

        if act_func == 'relu':
            self.act_func = nn.ReLU(inplace=True)
        else:
            self.act_func = Hswish(inplace=True)
        if attention:
            self.se = SELayer3D(interChannels)
        else:
            self.se = nn.Sequential()
        self.drop_connect_rate = drop_connect_rate
        self.stride = stride
        self.dense = int(dense)
        self.C_in = C_in
        self.C_out = C_out

    def forward(self, x):
        out = self.op1(x)
        out = self.act_func(out)
        out = self.op2(out)
        out = self.act_func(out)
        out = self.se(out)
        out = self.op3(out)

        if self.drop_connect_rate > 0:
            out = drop_path(out, drop_prob=self.drop_connect_rate, training=self.training)
        if self.stride == 1 and self.dense:
            out = torch.cat([x, out], dim=1)
        elif self.stride == 1 and self.C_in == self.C_out:
            out = out + x
        return out
class DenseBlock(nn.Module):
    def __init__(self, layer_types, in_channels, out_channels, kernel_sizes, expansions, strides, act_funcs, attentions,
                 drop_connect_rates, dense):
        super(DenseBlock, self).__init__()
        self.layer_types = list(map(int, layer_types.split()))
        self.in_channels = list(map(int, in_channels.split()))
        self.out_channels = list(map(int, out_channels.split()))
        self.kernel_sizes = list(map(int, kernel_sizes.split()))
        self.expansions = list(map(int, expansions.split()))
        self.attentions = list(map(bool, map(int, attentions.split())))
        self.strides = list(map(int, strides.split()))
        self.act_funcs = list(map(str, act_funcs.split()))
        self.drop_connect_rates = list(map(float, drop_connect_rates.split()))
        self.dense = int(dense)

        self.layer = self._make_dense(len(self.out_channels))

    def _make_dense(self, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            if self.layer_types[i] == 0:
                layers.append(Identity())
            else:
                layers.append(
                    MBConv(self.in_channels[i], self.out_channels[i], self.kernel_sizes[i], self.expansions[i],
                           self.strides[i], 1, self.act_funcs[i], self.attentions[i], self.drop_connect_rates[i],
                           self.dense))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.layer(out)
        return out
class DenseBlock3D(nn.Module):
    def __init__(self, layer_types, in_channels, out_channels, kernel_sizes, expansions, strides, act_funcs, attentions,
                 drop_connect_rates, dense):
        super(DenseBlock3D, self).__init__()
        self.layer_types = list(map(int, layer_types.split()))
        self.in_channels = list(map(int, in_channels.split()))
        self.out_channels = list(map(int, out_channels.split()))
        self.kernel_sizes = list(map(int, kernel_sizes.split()))
        self.expansions = list(map(int, expansions.split()))
        self.attentions = list(map(bool, map(int, attentions.split())))
        self.strides = list(map(int, strides.split()))
        self.act_funcs = list(map(str, act_funcs.split()))
        self.drop_connect_rates = list(map(float, drop_connect_rates.split()))
        self.dense = int(dense)

        self.layer = self._make_dense(len(self.out_channels))

    def _make_dense(self, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            if self.layer_types[i] == 0:
                layers.append(Identity())
            else:
                layers.append(
                    MBConv3D(self.in_channels[i], self.out_channels[i], self.kernel_sizes[i], self.expansions[i],
                           self.strides[i], 1, self.act_funcs[i], self.attentions[i], self.drop_connect_rates[i],
                           self.dense))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.layer(out)
        return out
class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.Hswish = Hswish(inplace=True)
        #generated_init

    def forward(self, x):
        #generate_forward
        dataset_name = Utils.get_dataset_name('NETWORK', 'name')
        
        out = self.evolved_block2(out)
        # out = self.Hswish(self.bn_end1(self.conv_end1(out)))
        if dataset_name == 'OrganMNIST3D'or dataset_name == 'NoduleMNIST3D'or dataset_name == 'AdrenalMNIST3D'or dataset_name == 'FractureMNIST3D'or dataset_name == 'VesselMNIST3D'or dataset_name == 'SynapseMNIST3D':
            out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        else:
            out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.Hswish(self.conv_end1(out))
        out = out.view(out.size(0), -1)
        # out = self.Hswish(self.dropout(self.linear1(out)))

        out = F.dropout(out, p=0.2, training=self.training)

        logit = self.kan_linear(out)

        return out, logit
    def extract_features(self, x):
        layer_features = []

        # Initial Convolution Block
        output = self.Hswish(self.bn_begin1(self.conv_begin1(x)))
        
        # Evolved Block 0
        for block in self.evolved_block0.layer:
            output = block(output)
            if isinstance(block, MBConv) :
                layer_features.append(output)

        # Evolved Block 1
        for block in self.evolved_block1.layer:
            output = block(output)
            if isinstance(block, MBConv) :
                layer_features.append(output)

        # Evolved Block 2
        for block in self.evolved_block2.layer:
            output = block(output)
            if isinstance(block, MBConv):
                layer_features.append(output)

        return layer_features
class TrainModel(object):
    def __init__(self, is_test, particle, batch_size, weight_decay):
        agent_name = Utils.get_dataset_name('SEARCH', 'agent_name')
        if is_test:
            dataset_name = Utils.get_dataset_name('NETWORK', 'name')
            print("Final Train is Start..........batchsize:%d, dataset:%s" % (batch_size, dataset_name))
            full_trainloader = data_loader.get_train_loader(dataset_name, batch_size=batch_size,
                                                            shuffle=True, num_workers=4, pin_memory=True)
            testloader = data_loader.get_test_loader(dataset_name, batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=4, pin_memory=True)
            self.full_trainloader = full_trainloader
            self.testloader = testloader
        else:
            dataset_name = Utils.get_dataset_name('NETWORK', 'name')
            print("Searching is start..........batchsize: %d, dataset: %s" % (8, dataset_name))
            Etrain, Evalid = data_loader.get_train_valid_loader(dataset_name, batch_size=8, subset_size=1,
                        valid_size=0.1, shuffle=True, num_workers=4,
                        pin_memory=True)
            self.Etrain = Etrain
            self.Evalid = Evalid

        self.agent_name = agent_name
        cudnn.benchmark = True
        self.net = EvoCNNModel()
        self.criterion = nn.CrossEntropyLoss()
        self.best_acc = 0.0
        # agent
        self.agent = 0
        self.params = 0
        self.flops = 10e9
        self.best_epoch = 0
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.weight_decay = weight_decay
        self.particle = copy.deepcopy(particle)
        self.net = self.net.cuda()
        self.latence = 0
        dataset_name = Utils.get_dataset_name('NETWORK', 'name')
        if dataset_name == 'OrganMNIST3D'or dataset_name == 'NoduleMNIST3D'or dataset_name == 'AdrenalMNIST3D'or dataset_name == 'FractureMNIST3D'or dataset_name == 'VesselMNIST3D'or dataset_name == 'SynapseMNIST3D':
            self.is_3D = True
        else:
            self.is_3D = False

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt' % (self.file_id), file_mode)
        f.write('[%s]-%s\n' % (dt, _str))
        f.flush()
        f.close()

    # Architecture evaluation
    def NOMAL_TRAIN(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for ii, data in enumerate(self.Etrain, 0):
            inputs, labels = data
            labels = labels.view(-1)
            labels = labels.to(dtype=torch.long)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            _, outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
            if epoch == 0 and ii == 0:
                image_channel = Utils.get_params('NETWORK', 'image_channel')
                if self.is_3D:
                    inputs = torch.randn(1, image_channel, 28, 28, 28)
                else:
                    inputs = torch.randn(1, image_channel, 28, 28)
                inputs = Variable(inputs.cuda())
                self.params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                self.flops = profile_macs(copy.deepcopy(self.net), inputs)
                self.log_record('#parameters:%d, #FLOPs:%d' % (self.params, self.flops))
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f'% (epoch+1, running_loss/total, (correct/total)))
    def NOMAL_EVAL(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        is_terminate = 0
        for ii, data in enumerate(self.Evalid, 0):
            inputs, labels = data
            labels = labels.view(-1)
            labels = labels.to(dtype=torch.long)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            if epoch == 0 and ii == 0:
                # gpu = int(Utils.get_params('SEARCH', 'GPU'))
                # gpu
                self.net = self.net.cuda()
                inputs = inputs.cuda()
                labels = labels.cuda()
                start_timer = time.time()
                _, outputs = self.net(inputs)
                end_time = time.time()
                self.latence = end_time - start_timer
                self.log_record('Search latence:%.4f gpu '% (self.latence))
                # cpu
                self.net = self.net.to('cpu')
                inputs = inputs.to('cpu')
                labels = labels.to('cpu')
                start_timer = time.time()
                _, outputs = self.net(inputs)
                end_time = time.time()
                self.latence = end_time - start_timer
                self.log_record('Search latence:%.4f cpu' % (self.latence))

            self.net = self.net.cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()
            _, outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if epoch >= self.best_epoch + 4 or float(correct / total) - self.agent < -0.03:
            is_terminate = 1
        if correct / total > self.agent:
            self.best_epoch = epoch
            self.agent = float(correct / total)
        self.log_record('Validate-Epoch:%4d,  Validate-Loss:%.4f, Acc:%.4f'%(epoch + 1, test_loss/total, correct/total))
        return is_terminate    
    def Zen_Score(self):
        image_channel = Utils.get_params('NETWORK', 'image_channel')
        repeat = int(Utils.get_params('SEARCH', 'repeat'))
        # gpu
        start_timer = time.time()
        info = compute_ZENSCORE_score(model=self.net, resolution=28, batch_size=1, image_channel = image_channel, is_3D= self.is_3D, repeat=repeat, gpu = 1)
        self.agent = info['avg_nas_score']
        self.log_record('Gpu Search Latence : %4f second(s), zen-score: %4f' % (
            time.time() - start_timer, self.agent))
        # cpu
        start_timer2 = time.time()
        info = compute_ZENSCORE_score(model=self.net, resolution=28, batch_size=1, image_channel = image_channel, is_3D= self.is_3D, repeat=repeat, gpu = 0)
        self.log_record('Cpu Search Latence : %4f second(s), zen-score: %4f' % (
                    time.time() - start_timer2, self.agent))
        if self.is_3D:
            inputs = torch.randn(1, image_channel, 28, 28, 28)
        else:
            inputs = torch.randn(1, image_channel, 28, 28)
        
        self.net = self.net.cuda()
        inputs = Variable(inputs.cuda())
        self.params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.flops = profile_macs(copy.deepcopy(self.net), inputs)
        self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))
        
    
    def Fisher(self):
        image_channel = Utils.get_params('NETWORK', 'image_channel')
        repeat = int(Utils.get_params('SEARCH', 'repeat'))
        if self.is_3D:
            inputs = torch.randn(1, image_channel, 28, 28, 28)
        else:
            inputs = torch.randn(1, image_channel, 28, 28)
        inputs = Variable(inputs.cuda())
        self.params = sum(p.numel() for p in self.net.parameters())
        self.flops = profile_macs(copy.deepcopy(self.net), inputs)
        start_timer = time.time()
        sums = 0
        # gpu
        for _ in range(repeat):
            the_score = compute_fisher_score(model=self.net, resolution=28, batch_size=1, image_channel = image_channel, is_3D = self.is_3D, gpu=1)
            sums += the_score
        time_cost = (time.time() - start_timer)
        self.agent = sums / repeat
        self.log_record(f'fisher_SCORE={sums / repeat:.4g}, Gpu latence={time_cost:.4g} second(s) ')
        # cpu
        start_timer2 = time.time()
        for _ in range(repeat):
            the_score = compute_fisher_score(model=self.net, resolution=28, batch_size=1, image_channel = image_channel, is_3D = self.is_3D, gpu=0)
        time_cost2 = (time.time() - start_timer2)
        # self.agent = sums / repeat)
        self.log_record(f'fisher_SCORE={sums / repeat:.4g}, cpu latence={time_cost2:.4g} second(s) ')

        self.net = self.net.cuda()
        # print(f'fisher_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
        self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))


    def Flops(self):
            image_channel = Utils.get_params('NETWORK', 'image_channel')
            repeat = int(Utils.get_params('SEARCH', 'repeat'))
            if self.is_3D:
                inputs = torch.randn(1, image_channel, 28, 28, 28)
            else:
                inputs = torch.randn(1, image_channel, 28, 28)
            inputs = Variable(inputs.cuda())
            self.params = sum(p.numel() for p in self.net.parameters())
            self.flops = profile_macs(copy.deepcopy(self.net), inputs)
            start_timer = time.time()
            sums = 0
            # gpu
            for _ in range(repeat):
                the_score = compute_flops_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 1)
                sums += the_score
            time_cost = (time.time() - start_timer)
            self.agent = (sums / repeat)/1000000
            self.log_record(f'flops_SCORE={sums / repeat:.4g}, Gpu Latence={time_cost:.4g} second(s)')

            # cpu
            start_timer2 = time.time()
            for _ in range(repeat):
                the_score = compute_flops_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 0)
                # sums += the_score
            time_cost2 = (time.time() - start_timer2)
            # self.agent = (sums / repeat)/1000000
            self.log_record(f'flops_SCORE={sums / repeat:.4g}, Cpu Latence={time_cost2:.4g} second(s)')

            # print(f'flops_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
            self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))
            

    def Params(self):
            image_channel = Utils.get_params('NETWORK', 'image_channel')
            repeat = int(Utils.get_params('SEARCH', 'repeat'))
            if self.is_3D:
                inputs = torch.randn(1, image_channel, 28, 28, 28)
            else:
                inputs = torch.randn(1, image_channel, 28, 28)
            inputs = Variable(inputs.cuda())
            self.params = sum(p.numel() for p in self.net.parameters())
            self.flops = profile_macs(copy.deepcopy(self.net), inputs)
            start_timer = time.time()
            sums = 0
            # gpu
            for _ in range(repeat):
                the_score = compute_params_score(model=self.net, resolution=28, batch_size=1, image_channel = image_channel, is_3D = self.is_3D,gpu = 1)
                sums += the_score
            time_cost = (time.time() - start_timer)
            self.agent = (sums / repeat)/1000000
            self.log_record(f'params_SCORE={sums / repeat:.4g}, Gpu Latence={time_cost:.4g} second(s)')
            # gpu
            start_timer2 = time.time()
            for _ in range(repeat):
                the_score = compute_params_score(model=self.net, resolution=28, batch_size=1, image_channel = image_channel, is_3D = self.is_3D,gpu = 0)
                # sums += the_score
            time_cost2 = (time.time() - start_timer2)
            # self.agent = (sums / repeat)/1000000
            self.log_record(f'params_SCORE={sums / repeat:.4g}, Cpu Latence={time_cost2:.4g} second(s)')

            # print(f'params_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
            self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))


    def Gradnorm(self):
            image_channel = Utils.get_params('NETWORK', 'image_channel')
            repeat = int(Utils.get_params('SEARCH', 'repeat'))
            if self.is_3D:
                inputs = torch.randn(1, image_channel, 28, 28, 28)
            else:
                inputs = torch.randn(1, image_channel, 28, 28)
            inputs = Variable(inputs.cuda())
            self.params = sum(p.numel() for p in self.net.parameters())
            self.flops = profile_macs(copy.deepcopy(self.net), inputs)
            start_timer = time.time()
            sums = 0
            # gpu
            for _ in range(repeat):
                the_score = compute_GRADNORM_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 1)
                sums += the_score
            time_cost = (time.time() - start_timer)
            self.agent = float(sums / repeat)
            self.log_record(f'Gradnorm_SCORE={sums / repeat:.4g}, Gpu Latence={time_cost:.4g} second(s)')

            # cpu
            start_timer2 = time.time()
            for _ in range(repeat):
                the_score = compute_GRADNORM_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 0)
                # sums += the_score
            time_cost2 = (time.time() - start_timer2)
            # self.agent = float(sums / repeat)
            self.log_record(f'Gradnorm_SCORE={sums / repeat:.4g}, Cpu Latence={time_cost2:.4g} second(s)')

            # print(f'Gradnorm_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
            self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))
            

    def Grasp(self):
            image_channel = Utils.get_params('NETWORK', 'image_channel')
            repeat = int(Utils.get_params('SEARCH', 'repeat'))
            if self.is_3D:
                inputs = torch.randn(1, image_channel, 28, 28, 28)
            else:
                inputs = torch.randn(1, image_channel, 28, 28)
            inputs = Variable(inputs.cuda())
            self.params = sum(p.numel() for p in self.net.parameters())
            self.flops = profile_macs(copy.deepcopy(self.net), inputs)
            start_timer = time.time()
            sums = 0
            # gpu
            for _ in range(repeat):
                the_score = compute_grasp_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 1)
                sums += the_score
            time_cost = (time.time() - start_timer)
            self.agent = float(sums / repeat)
            self.log_record(f'Grasp_SCORE={sums / repeat:.4g}, Gpu Latence={time_cost:.4g} second(s)')
            # cpu
            start_timer2 = time.time()
            for _ in range(repeat):
                the_score = compute_grasp_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 0)
                # sums += the_score
            time_cost2 = (time.time() - start_timer2)
            # self.agent = float(sums / repeat)
            self.log_record(f'Grasp_SCORE={sums / repeat:.4g}, Cpu Latence={time_cost2:.4g} second(s)')
            # print(f'Grasp_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
            self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))
            

    def Naswot(self):
            image_channel = Utils.get_params('NETWORK', 'image_channel')
            repeat = int(Utils.get_params('SEARCH', 'repeat'))
            if self.is_3D:
                inputs = torch.randn(1, image_channel, 28, 28, 28)
            else:
                inputs = torch.randn(1, image_channel, 28, 28)
            inputs = Variable(inputs.cuda())
            self.params = sum(p.numel() for p in self.net.parameters())
            self.flops = profile_macs(copy.deepcopy(self.net), inputs)
            start_timer = time.time()
            sums = 0
            # gpu
            for _ in range(repeat):
                the_score = compute_NASWOT_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 1)
                sums += the_score
            time_cost = (time.time() - start_timer)
            self.agent = float(sums / repeat)
            self.log_record(f'NASWOT_SCORE={sums / repeat:.4g}, Gpu Latence={time_cost:.4g} second(s)')

            # cpu
            start_timer2 = time.time()
            for _ in range(repeat):
                the_score = compute_NASWOT_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 0)
                # sums += the_score
            time_cost2 = (time.time() - start_timer2)
            # self.agent = float(sums / repeat)
            self.log_record(f'NASWOT_SCORE={sums / repeat:.4g}, Cpu Latence={time_cost2:.4g} second(s)')

            # print(f'NASWOT_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
            self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))
            

    def Ntk(self):
            image_channel = Utils.get_params('NETWORK', 'image_channel')
            repeat = int(Utils.get_params('SEARCH', 'repeat'))
            if self.is_3D:
                inputs = torch.randn(1, image_channel, 28, 28, 28)
            else:
                inputs = torch.randn(1, image_channel, 28, 28)
            inputs = Variable(inputs.cuda())
            self.params = sum(p.numel() for p in self.net.parameters())
            self.flops = profile_macs(copy.deepcopy(self.net), inputs)
            start_timer = time.time()
            sums = 0
            # gpu
            for _ in range(repeat):
                the_score = compute_NTK_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 1)
                sums += the_score
            time_cost = (time.time() - start_timer)
            self.agent = float(sums / repeat)
            self.log_record(f'NTK_SCORE={sums / repeat:.4g}, Gpu Latence={time_cost:.4g} second(s)')

            # cpu
            start_timer2 = time.time()
            for _ in range(repeat):
                the_score = compute_NTK_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 0)
                # sums += the_score
            time_cost2 = (time.time() - start_timer2)
            # self.agent = float(sums / repeat)
            self.log_record(f'NTK_SCORE={sums / repeat:.4g}, Cpu Latence={time_cost2:.4g} second(s)')

            # print(f'NTK_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
            self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))
            

    def Snip(self):
            image_channel = Utils.get_params('NETWORK', 'image_channel')
            repeat = int(Utils.get_params('SEARCH', 'repeat'))
            if self.is_3D:
                inputs = torch.randn(1, image_channel, 28, 28, 28)
            else:
                inputs = torch.randn(1, image_channel, 28, 28)
            inputs = Variable(inputs.cuda())
            self.params = sum(p.numel() for p in self.net.parameters())
            self.flops = profile_macs(copy.deepcopy(self.net), inputs)
            start_timer = time.time()
            sums = 0
            # gpu
            for _ in range(repeat):
                the_score = compute_snip_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 1)
                sums += the_score
            time_cost = (time.time() - start_timer)
            self.agent = float(sums / repeat)
            self.log_record(f'Snip_SCORE={sums / repeat:.4g}, Gpu Latence={time_cost:.4g} second(s)')
            # print(f'Snip_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
            # cpu
            start_timer2 = time.time()
            for _ in range(repeat):
                the_score = compute_snip_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 0)
                # sums += the_score
            time_cost2 = (time.time() - start_timer2)
            # self.agent = float(sums / repeat)
            self.log_record(f'Snip_SCORE={sums / repeat:.4g}, Cpu Latence={time_cost2:.4g} second(s)')
            self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))
            


    def Syncflow(self):
            image_channel = Utils.get_params('NETWORK', 'image_channel')
            repeat = int(Utils.get_params('SEARCH', 'repeat'))
            if self.is_3D:
                inputs = torch.randn(1, image_channel, 28, 28, 28)
            else:
                inputs = torch.randn(1, image_channel, 28, 28)
            inputs = Variable(inputs.cuda())
            self.params = sum(p.numel() for p in self.net.parameters())
            self.flops = profile_macs(copy.deepcopy(self.net), inputs)
            start_timer = time.time()
            sums = 0
            # gpu
            for _ in range(repeat):
                the_score = compute_SYNCFLOW_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 1)
                sums += the_score
            time_cost = (time.time() - start_timer)
            self.agent = float(sums / repeat)
            self.log_record(f'Syncflow_SCORE={sums / repeat:.4g}, Gpu Latence={time_cost:.4g} second(s)')

            # cpu
            start_timer2 = time.time()
            for _ in range(repeat):
                the_score = compute_SYNCFLOW_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 0)
                # sums += the_score
            time_cost2 = (time.time() - start_timer2)
            # self.agent = float(sums / repeat)
            self.log_record(f'Syncflow_SCORE={sums / repeat:.4g}, Cpu Latence={time_cost2:.4g} second(s)')

            # print(f'Syncflow_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
            self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))
            
    def Zico(self):
        image_channel = Utils.get_params('NETWORK', 'image_channel')
        repeat = int(Utils.get_params('SEARCH', 'repeat'))
        if self.is_3D:
            inputs = torch.randn(1, image_channel, 28, 28, 28)
        else:
            inputs = torch.randn(1, image_channel, 28, 28)
        inputs = Variable(inputs.cuda())
        self.params = sum(p.numel() for p in self.net.parameters())
        self.flops = profile_macs(copy.deepcopy(self.net), inputs)
        start_timer = time.time()
        sums = 0
        # gpu
        for _ in range(repeat):
            the_score = compute_zico_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 1)
            sums += the_score
        time_cost = (time.time() - start_timer)
        self.agent = float(sums / repeat)
        self.log_record(f'Zico_SCORE={sums / repeat:.4g}, Gpu Latence={time_cost:.4g} second(s)')

        # cpu
        start_timer2 = time.time()
        for _ in range(repeat):
            the_score = compute_zico_score(model=self.net, resolution=28, batch_size=1, image_channel=image_channel, is_3D=self.is_3D, gpu = 0)
            # sums += the_score
        time_cost2 = (time.time() - start_timer2)
        # self.agent = float(sums / repeat)
        self.log_record(f'Zico_SCORE={sums / repeat:.4g}, Cpu Latence={time_cost2:.4g} second(s)')

        # print(f'Syncflow_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
        self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))
    def My(self):
        image_channel = Utils.get_params('NETWORK', 'image_channel')
        repeat = int(Utils.get_params('SEARCH', 'repeat'))
        if self.is_3D:
            inputs = torch.randn(1, image_channel, 28, 28, 28)
        else:
            inputs = torch.randn(1, image_channel, 28, 28)
        inputs = Variable(inputs.cuda())
        self.params = sum(p.numel() for p in self.net.parameters())
        self.flops = profile_macs(copy.deepcopy(self.net), inputs)
        start_timer = time.time()
        # gpu
        the_scores = compute_our_score(model=self.net, resolution=28, image_channel=image_channel, is_3D=self.is_3D, gpu = 1)
        time_cost = (time.time() - start_timer)
        
        beta = 0.1  # Factor controlling the penalization strength
        c_f = 50  # Ideal FLOPs in millions
        c_p = 2  # Ideal parameters in millions
        alpha = 2  # Linear scaling factor for expressivity

        # Step 1: Balanced complexity with Gaussian penalization
        balanced_complexity_flops = np.exp(-beta * (the_scores['complexity_flops'] / 1e6 - c_f) ** 2)
        balanced_complexity_params = np.exp(-beta * (the_scores['complexity_params'] / 1e6 - c_p) ** 2)

        # Combine the two complexities
        balanced_complexity = balanced_complexity_flops * balanced_complexity_params

        # Step 2: Linearly scale expressivity
        scaled_expressivity = 1 + alpha * the_scores['expressivity']

        # Step 3: Combine all metrics using multiplication
        final_score_balanced = scaled_expressivity * the_scores['trainability'] / (1 + balanced_complexity)
        
        # values = np.abs([the_scores['expressivity'], the_scores['trainability'], the_scores['complexity_flops']/1000000, the_scores['complexity_params']/1000000, time_cost])
        # log_values = np.log1p(values)
        # combined_value = np.exp(np.mean(log_values))
        
        
        self.agent = float(final_score_balanced)
        self.log_record(f'my_SCORE={final_score_balanced:.4g}, Gpu Latence={time_cost:.4g} second(s)')

        # cpu
        start_timer2 = time.time()
        the_score = compute_our_score(model=self.net, resolution=28, image_channel=image_channel, is_3D=self.is_3D, gpu = 0)
        time_cost2 = (time.time() - start_timer2)
        self.log_record(f'my_SCORE={final_score_balanced:.4g}, Cpu Latence={time_cost2:.4g} second(s)')

        # print(f'Syncflow_SCORE={sums / repeat:.4g}, time cost={time_cost:.4g} second(s)')
        self.log_record('#Parameters:%d, #FLOPs:%d' % (self.params, self.flops))
    def process(self):
        if self.agent_name == 'ZEN_SCORE':
            self.Zen_Score()
            return self.agent, self.params, self.flops
        elif self.agent_name == 'NORMAL':
            min_epoch_eval = Utils.get_params('NETWORK', 'min_epoch_eval')
            lr_rate = 0.08
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr_rate, momentum=0.9, weight_decay=4e-5, nesterov=True)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, min_epoch_eval)
            is_terminate = 0
            latences = 0
            count = 0
            for p in range(min_epoch_eval):
                if not is_terminate:
                    count += 1
                    start = time.time()
                    self.NOMAL_TRAIN(p, optimizer)
                    scheduler.step()
                    is_terminate = self.NOMAL_EVAL(p)
                    latences += time.time() - start
                else:
                    return self.agent, self.params, self.flops
            self.log_record('#Latence:%d' % (float(latences / count)))
            return self.agent, self.params, self.flops
        elif self.agent_name == 'FISHER':
            self.Fisher()
            # print(float(self.agent))

            return self.agent, self.params, self.flops
        elif self.agent_name == 'FLOPS':
            self.Flops()
            return self.agent, self.params, self.flops
        elif self.agent_name == 'PARAMS':
            self.Params()
            return self.agent, self.params, self.flops
            # pass
        elif self.agent_name == 'GRADNORM':
            # pass
            self.Gradnorm()
            return self.agent, self.params, self.flops
        elif self.agent_name == 'GRASP':
            # pass
            self.Grasp()
            return self.agent, self.params, self.flops
        elif self.agent_name == 'NASWOT':
            # pass
            self.Naswot()
            return self.agent, self.params, self.flops
        elif self.agent_name == 'NTK':
            # pass
            self.Ntk()
            return self.agent, self.params, self.flops
        elif self.agent_name == 'SNIP':
            self.Snip()
            return self.agent, self.params, self.flops
        elif self.agent_name == 'SYNCFLOW':
            # pass
            self.Syncflow()
            return self.agent, self.params, self.flops
        elif self.agent_name == 'ZICO':
            self.Zico()
            return self.agent, self.params, self.flops
        elif self.agent_name == 'MY':
            self.My()
            return self.agent, self.params, self.flops
    # final train test
    def final_train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        # flops = 10e9
        full_trainloader = tqdm(self.full_trainloader, file=sys.stdout, leave=True)
        gpu_latence = 0
        cpu_latence = 0
        for ii, data in enumerate(full_trainloader, 0):

            inputs, labels = data
            labels = labels.view(-1)
            labels = labels.to(dtype=torch.long)
            # inputs = F.interpolate(inputs, size=40, mode='bicubic', align_corners=False)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            if epoch == 0:
                # gpu
                self.net = self.net.cuda()
                inputs = inputs.cuda()
                labels = labels.cuda()
                start_timer = time.time()
                _, outputs = self.net(inputs)
                end_time = time.time()
                self.latence = end_time - start_timer
                gpu_latence = self.latence + gpu_latence
                
                # cpu
                self.net = self.net.to('cpu')
                inputs = inputs.to('cpu')
                labels = labels.to('cpu')
                start_timer = time.time()
                _, outputs = self.net(inputs)
                end_time = time.time()
                self.latence = end_time - start_timer
                cpu_latence = self.latence + cpu_latence
            if ii == len(full_trainloader) - 1 and epoch == 0:
                self.log_record('Train latence:%.4f gpu '% (gpu_latence))
                self.log_record('Train latence:%.4f cpu' % (cpu_latence))

            self.net = self.net.cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()
            _, outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

            full_trainloader.desc = "[Train epoch {}]  Loss {} Acc {}".format(epoch + 1, running_loss / total,
                                                                        correct / total)

            if epoch == 0 and ii == 0:
                image_channel = Utils.get_params('NETWORK', 'image_channel')
                if self.is_3D:
                    inputs = torch.randn(1, image_channel, 28, 28, 28)
                else:
                    inputs = torch.randn(1, image_channel, 28, 28)
                # inputs = torch.randn(1, image_channel, 28, 28)
                inputs = Variable(inputs.cuda())
                self.net = self.net.cuda()
                self.params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                # flops1, params1 = profile(self.net, inputs=(inputs,))
                self.flops = profile_macs(copy.deepcopy(self.net), inputs)
                self.log_record('#parameters:%d, #FLOPs:%d' % (self.params, self.flops))
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f' % (epoch + 1, running_loss / total, (correct / total)))
    def final_test(self, epoch, dataset_name): 
        with torch.no_grad():
            self.net.eval()
            test_loss = 0.0
            total = 0
            correct = 0

            all_predictions = []
            all_labels = []
            testloader = tqdm(self.testloader, file=sys.stdout, leave=True)
            for _, data in enumerate(testloader, 0):
                inputs, labels = data
                labels = labels.view(-1)
                labels = labels.to(dtype=torch.long)

                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                _, outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.data).sum()

                outputs = F.softmax(outputs, dim=1)
                all_predictions.extend(outputs.cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())

                testloader.desc = "[Eval epoch {}]  Loss {} Acc {} ".format(epoch + 1, test_loss / total,
                                                                            correct / total)
        if dataset_name == 'PneumoniaMNIST' or dataset_name == 'BreastMNIST' or dataset_name == 'NoduleMNIST3D' or dataset_name == 'AdrenalMNIST3D' or dataset_name == 'VesselMNIST3D' or dataset_name == 'SynapseMNIST3D':
            roc_auc = roc_auc_score(all_labels, np.argmax(all_predictions, axis=1))
        else:
            roc_auc = roc_auc_score(all_labels, all_predictions, multi_class='ovo', average='weighted')
        cohen_kappa = cohen_kappa_score(all_labels, np.argmax(all_predictions, axis=1))
        f1 = f1_score(all_labels, np.argmax(all_predictions, axis=1), average='weighted')
        IBA = balanced_accuracy_score(all_labels, np.argmax(all_predictions, axis=1))

        if correct / total > self.best_acc:
            torch.save(self.net.state_dict(), './trained_models/best_CNN.pt')
            self.best_acc = correct / total

        self.log_record(
            'Test-Loss:%.4f, Acc:%.4f, cohen_kappa:%.4f, ROC-AUC:%.4f, F1-Score:%.4f, IBA:%.4f' % (
                test_loss / total, correct / total, cohen_kappa, roc_auc, f1, IBA))
    def process_test(self):
        # params = sum(
        #     p.numel() for n, p in self.net.named_parameters() if p.requires_grad and not n.__contains__('auxiliary'))
        total_epoch = Utils.get_params('SEARCH', 'epoch_test')
        dataset_name = Utils.get_dataset_name('NETWORK', 'name')
        lr_rate = 0.0007
        optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

        # self.test(0)
        for p in range(total_epoch):
            if p < 5:
                optimizer_ini = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=self.weight_decay)
                self.final_train(p, optimizer_ini)
                self.final_test(p,dataset_name)
            else:
                self.final_train(p, optimizer)
                self.final_test(p,dataset_name)
                scheduler.step()
        return self.best_acc, self.params, self.flops


class RunModel(object):
    def do_work(self, gpu_id, curr_gen, file_id, is_test, particle=None, batch_size=None, weight_decay=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        params = 1e9
        flops = 10e9
        agent = 0

        m = TrainModel(is_test, particle, batch_size, weight_decay)
        m.log_record(
            'Used GPU#%s, worker name:%s[%d]' % (gpu_id, multiprocessing.current_process().name, os.getpid()))
        if is_test:
            best_acc, params, flops = m.process_test()
        else:
            agent, params, flops = m.process()
            # print(agent)

        if is_test:
            m.log_record('Finished-Acc:%.4f' % best_acc)
            m.log_record('Finished-Err:%.4f' % (1 - best_acc))
        else:
            agent_name = Utils.get_dataset_name('SEARCH', 'agent_name')
            m.log_record('%s is: %s' % (agent_name, str(agent)))

        f1 = open('./populations/agent_%02d.txt' % (curr_gen), 'a+')
        if is_test:
            f1.write('%s=%.5f\n' % (file_id, 1 - best_acc))
        else:
            f1.write('%s=%s\n' % (file_id, str(agent)))
        f1.flush()
        f1.close()

        f2 = open('./populations/params_%02d.txt' % (curr_gen), 'a+')
        f2.write('%s=%d\n' % (file_id, params))
        f2.flush()
        f2.close()

        f3 = open('./populations/flops_%02d.txt' % (curr_gen), 'a+')
        f3.write('%s=%d\n' % (file_id, flops))
        f3.flush()
        f3.close()
"""