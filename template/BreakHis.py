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
import load_dataset.data_loader_BreakHis as data_loader
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
from compute_zen_score import compute_nas_score
from tqdm import tqdm
import sys

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


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.Hswish = Hswish(inplace=True)
        #generated_init

    def forward(self, x):
        #generate_forward

        out = self.evolved_block2(out)
        # out = self.Hswish(self.bn_end1(self.conv_end1(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.Hswish(self.conv_end1(out))
        out = out.view(out.size(0), -1)
        # out = self.Hswish(self.dropout(self.linear1(out)))

        out = F.dropout(out, p=0.2, training=self.training)
        out = self.linear(out)

        return out


class TrainModel(object):
    def __init__(self, is_test, particle, batch_size, weight_decay):
        if is_test:
            print("final train is start..........batchsize:%d " % (batch_size))
            full_trainloader = data_loader.get_train_loader('../datasets/BreakHis_data/train', batch_size=batch_size,
                                                            augment=True, shuffle=True, num_workers=4, pin_memory=True)
            testloader = data_loader.get_test_loader('../datasets/BreakHis_data/val', batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=4, pin_memory=True)
            self.full_trainloader = full_trainloader
            self.testloader = testloader

        cudnn.benchmark = True
        self.net = EvoCNNModel()
        self.criterion = nn.CrossEntropyLoss()
        self.best_acc = 0.0
        self.zen_score = 0
        self.params = 0
        self.flops = 10e9
        self.best_epoch = 0
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.weight_decay = weight_decay
        self.particle = copy.deepcopy(particle)
        self.net = self.net.cuda()

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
    def train(self):
        start_timer = time.time()
        info = compute_nas_score(model=self.net, resolution=256, batch_size=8)
        time_cost = (time.time() - start_timer) / 32
        self.zen_score = info['avg_nas_score']
        print(f'zen-score={self.zen_score:.4g}, time cost={time_cost:.4g} second(s)')

        inputs = torch.randn(1, 3, 32, 32)
        inputs = Variable(inputs.cuda())
        self.params = sum(p.numel() for p in self.net.parameters())
        self.flops = profile_macs(copy.deepcopy(self.net), inputs)
        self.log_record('#parameters:%d, #FLOPs:%d' % (self.params, self.flops))
        self.log_record('Evaluate Total time: %4f second(s), Avg Time cost: %4f second(s), zen-score: %4f' % (
            time.time() - start_timer, time_cost, self.zen_score))

    def process(self):
        self.train()
        return self.zen_score, self.params, self.flops

    # final train test
    def final_train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        # flops = 10e9
        full_trainloader = tqdm(self.full_trainloader, file=sys.stdout, leave=True)
        for ii, data in enumerate(full_trainloader, 0):
            inputs, labels = data
            # inputs = F.interpolate(inputs, size=40, mode='bicubic', align_corners=False)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
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
                inputs = torch.randn(1, 3, 32, 32)
                inputs = Variable(inputs.cuda())
                self.params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                # flops1, params1 = profile(self.net, inputs=(inputs,))
                self.flops = profile_macs(copy.deepcopy(self.net), inputs)
                self.log_record('#parameters:%d, #FLOPs:%d' % (self.params, self.flops))
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f' % (epoch + 1, running_loss / total, (correct / total)))

    def process_test(self):
        params = sum(
            p.numel() for n, p in self.net.named_parameters() if p.requires_grad and not n.__contains__('auxiliary'))
        total_epoch = Utils.get_params('SEARCH', 'epoch_test')
        lr_rate = 0.0001
        optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

        # self.test(0)
        for p in range(total_epoch):
            if p < 5:
                optimizer_ini = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=self.weight_decay)
                self.final_train(p, optimizer_ini)
                self.test(p)
            else:
                self.final_train(p, optimizer)
                self.test(p)
                scheduler.step()
        return self.best_acc, self.params, self.flops

    def test(self, epoch):
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
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                outputs = self.net(inputs)

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


class RunModel(object):
    def do_work(self, gpu_id, curr_gen, file_id, is_test, particle=None, batch_size=None, weight_decay=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        params = 1e9
        flops = 10e9
        zen_score = 0
        try:
            m = TrainModel(is_test, particle, batch_size, weight_decay)
            m.log_record(
                'Used GPU#%s, worker name:%s[%d]' % (gpu_id, multiprocessing.current_process().name, os.getpid()))
            if is_test:
                best_acc, params, flops = m.process_test()
            else:
                zen_score, params, flops = m.process()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s' % (file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s' % (str(e)))
        finally:
            if is_test:
                m.log_record('Finished-Acc:%.4f' % best_acc)
                m.log_record('Finished-Err:%.4f' % (1 - best_acc))
            else:
                m.log_record('zen_score:%.4f' % zen_score)

            f1 = open('./populations/zen_score_%02d.txt' % (curr_gen), 'a+')
            if is_test:
                f1.write('%s=%.5f\n' % (file_id, 1 - best_acc))
            else:
                f1.write('%s=%.5f\n' % (file_id, zen_score))
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