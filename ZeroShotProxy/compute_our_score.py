import copy
import os, sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
# from model_test_3D import EvoCNNModel as EvoCNNModel3D
# from model_test_2D import EvoCNNModel as EvoCNNModel2D
from torch.autograd import Variable
from torchprofile import profile_macs
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm3d)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm3d)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    else:
        raise NotImplementedError
    return model
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

def caculate_grad(grad_dict):
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

def most_frequent_element(arr):
    # 统计每个元素的出现次数
    count = Counter(arr)
    # 找出最大频率
    max_freq = max(count.values())
    # 找出所有出现次数等于最大频率的元素
    most_frequent = [k for k, v in count.items() if v == max_freq]
    return most_frequent
def compute_our_score(model, gpu, resolution, is_3D, image_channel, init=True, mixup_gamma = 1e-2):
    model.train()
    info = {}
    
    if gpu:
        device = torch.device('cuda:{}'.format(0))
        model = model.cuda()
    else:
        device = torch.device('cpu')
        model = model.cpu()

    if init:
        init_model(model, 'kaiming_norm_fanin')
    # if is_3D:
    #     inputs = torch.randn(1, image_channel, resolution, resolution, resolution, device=device)
    # else:
    #     inputs = torch.randn(1, image_channel, resolution, resolution, device=device)
    # input_ = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
    if is_3D:
        input = torch.randn(size=[1, image_channel, resolution, resolution, resolution], device=device)
        input2 = torch.randn(size=[1, image_channel, resolution, resolution, resolution], device=device)
    else:
        input = torch.randn(size=[1, image_channel, resolution, resolution], device=device)
        input2 = torch.randn(size=[1, image_channel, resolution, resolution], device=device)
    layer_features = model.extract_features(input)

    ################ genneration scores ################
    
    grad_dict= {}
    nas_score_list = []
    # data = inputs
    latences = []
    e1_latences = []
    e2_latences = []
    for i in range(16):
        start_e = time.time()
        if is_3D:
            input = torch.randn(size=[1, image_channel, resolution, resolution, resolution], device=device)
            input2 = torch.randn(size=[1, image_channel, resolution, resolution, resolution], device=device)
        else:
            input = torch.randn(size=[1, image_channel, resolution, resolution], device=device)
            input2 = torch.randn(size=[1, image_channel, resolution, resolution], device=device)
        mixup_input = input + mixup_gamma * input2
        start_time = time.time()
        out, logits1 = model(input)
        latences.append(round(time.time() - start_time, 2))
        mix_output, logits2 = model(mixup_input)
        nas_score = torch.sum(torch.abs(out - mix_output))
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
        avg_nas_score = np.mean(nas_score_list)
        e1_latences.append(round(time.time() - start_e, 2))
        # zico
        num_classes = logits1.shape[1]
        y = torch.randint(low=0, high=num_classes, size=[1]).cuda()
        one_hot_y = F.one_hot(y, num_classes).float()
        if gpu:
            one_hot_y = one_hot_y.cuda()
        else:
            one_hot_y = one_hot_y.cpu()
        loss = F.cross_entropy(logits1, one_hot_y)
        loss.backward()
        grad_dict= getgrad(model, grad_dict, i)
        e2_latences.append(round(time.time() - start_e, 2))
        
        # print('grad_dict: ', grad_dict)
    # latence = float(latences / 16)
    latence = min(most_frequent_element(latences))
    # e2_latence = float(e2_latences / 16)
    e2_latence = min(most_frequent_element(e2_latences))
    # e1_latence = float(e1_latences / 16)
    e1_latence = min(most_frequent_element(e1_latences))
    # print('latence: ', latence)
    # print('e1_latence: ', e1_latence)
    # print('e2_latence: ', e2_latence)
    expressivity = caculate_grad(grad_dict) * (1 - e2_latence) + avg_nas_score * (1 - e1_latence)

    #####################################################################

    ################ trainability score ##############
    scores = []
    # 反向遍历层特征，从最后一层开始
    for i in reversed(range(0, len(layer_features))):
        f_out = layer_features[i] # 当前层的输出
        f_in = layer_features[i-1] # 前一层的输入
        # 重置梯度
        if f_out.grad is not None:
            f_out.grad.zero_()
        if f_in.grad is not None:
            f_in.grad.zero_()
        
        # 使用不同分布的随机向量生成方法以增强估计
        # 这里使用高斯分布代替 Rademacher 向量
        g_out = torch.randn_like(f_out)
        g_out = (g_out - g_out.mean()) / g_out.std()  # 标准化随机向量
        # g_out = torch.ones_like(f_out) * 0.5
        # g_out = (torch.bernoulli(g_out) - 0.5) * 2
        # 计算输入和输出的梯度
        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out, retain_graph=False, allow_unused=True)[0]
        # 检查尺寸是否匹配，或随机梯度是否完全相等
        if g_in is not None and g_out is not None:

            if g_out.size()==g_in.size() and torch.all(g_in == g_out):
                scores.append(-np.inf)
            else:
                # 对不匹配的梯度调整大小
                if is_3D:
                    if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3) or g_out.size(4) != g_in.size(4):
                        bo, co, do, ho, wo = g_out.size()
                        bi, ci, di, hi, wi = g_in.size()
                        stride = int(di / do)
                        pixel_unshuffle = nn.PixelUnshuffle(stride)
                        g_in = pixel_unshuffle(g_in)
                        # 调整梯度矩阵的形状
                    bo, co, do, ho, wo = g_out.size()
                    bi, ci, di, hi, wi = g_in.size()
                    ### straight-forward way
                    # g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,1,co)
                    # g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci,1)
                    # mat = torch.bmm(g_in,g_out).mean(dim=0)
                    ### efficient way # print(torch.allclose(mat, mat2, atol=1e-6))
                    # 更高效的矩阵计算
                    g_out = g_out.permute(0,2,3,4,1).contiguous().view(bo*ho*wo*do,co)
                    g_in = g_in.permute(0,2,3,4,1).contiguous().view(bi*hi*wi*do,ci)
                    # 计算梯度矩阵的内积，使用更稳定的归一化方式
                    mat = torch.mm(g_in.transpose(1,0), g_out) / (bo*ho*wo*do)
                else:
                    if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3):
                        bo,co,ho,wo = g_out.size()
                        bi,ci,hi,wi = g_in.size()
                        stride = int(hi/ho)
                        pixel_unshuffle = nn.PixelUnshuffle(stride)
                        g_in = pixel_unshuffle(g_in)
                    # 调整梯度矩阵的形状
                    bo,co,ho,wo = g_out.size()
                    bi,ci,hi,wi = g_in.size()
                    ### straight-forward way
                    # g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,1,co)
                    # g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci,1)
                    # mat = torch.bmm(g_in,g_out).mean(dim=0)
                    ### efficient way # print(torch.allclose(mat, mat2, atol=1e-6))
                    # 更高效的矩阵计算
                    g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,co)
                    g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci)
                    # 计算梯度矩阵的内积，使用更稳定的归一化方式
                    mat = torch.mm(g_in.transpose(1,0), g_out) / (bo*ho*wo)
                ### make faster on cpu
                # 增加对矩阵的形状调整，优化 CPU 计算
                if mat.size(0) < mat.size(1):
                    mat = mat.transpose(0,1)
                ###
                # 计算奇异值，并对大值和小值做处理以提高数值稳定性

                s = torch.linalg.svdvals(mat)
                max_s = s.max().item()
                max_s = max(max_s, 1e-6)  # 防止数值过小导致的不稳定
                score = -max_s - 1 / (max_s + 1e-6) + 2
                # 为奇异值较大的情况增加正则化项，避免训练不稳定
                regularization = 1e-3 * (s - max_s).sum().item()
                scores.append(score + regularization)
                # scores.append(-s.max().item() - 1/(s.max().item()+1e-6)+2)
    trainability = abs(np.mean(scores) * (1 - latence))
    # print('trainability', trainability)
    #################################################
    #################### complexity #################
    # only Gpu
    inputs = Variable(input.cuda())
    model = model.cuda()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = profile_macs(copy.deepcopy(model), inputs)


    info['expressivity'] = float(expressivity) if not np.isnan(expressivity) else -np.inf
    info['trainability'] = float(trainability) if not np.isnan(trainability) else -np.inf
    info['complexity_flops'] = float(flops) if not np.isnan(expressivity) else -np.inf
    info['complexity_params'] = float(params) if not np.isnan(expressivity) else -np.inf
    return info
