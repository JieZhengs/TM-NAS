# !/usr/bin/python
# -*- coding: utf-8 -*-
import configparser
import logging
import time
import sys
import os
from subprocess import Popen, PIPE

# 配置对应表
kernel_sizes = [0, 3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 5, 5]
expansion_rates = [0, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6]
attent = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


class Utils(object):
    # 获取初始化参数
    @classmethod
    def get_init_params(cls):
        params = {}
        params['pop_size'] = cls.get_params('PSO', 'pop_size')
        params['num_iteration'] = cls.get_params('PSO', 'num_iteration')
        params['particle_length'] = cls.get_params('PSO', 'particle_length')
        params['max_strided'] = cls.get_params('NETWORK', 'max_strided')
        params['image_channel'] = cls.get_params('NETWORK', 'image_channel')
        params['max_output_channel'] = cls.get_params('NETWORK', 'max_output_channel')
        params['epoch_test'] = cls.get_params('SEARCH', 'epoch_test')
        return params

    # 获取初始化的配置文件global.ini
    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('global.ini')
        return config.get(section, key)

    @classmethod
    def get_params(cls, domain, key):
        rs = cls.__read_ini_file(domain, key)
        return int(rs)
    
    @classmethod
    def get_dataset_name(cls, domain, key):
        rs = cls.__read_ini_file(domain, key)
        return str(rs)

    # 生成的结果字符串直接保存成文件
    @classmethod
    def save_population_and_agent(cls, type, population, agent_set, num_parameters, flops, gen_no,
                                      proxy_err=None,
                                      time=None):
        file_name = './populations/' + type + '_%02d.txt' % (gen_no)
        _str = cls.popAndErr2str(population, agent_set, num_parameters, flops, proxy_err, time)
        with open(file_name, 'w') as f:
            f.write(_str)

    # 字符串写到log文件里
    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()

    # 把种群、错误率等结果信息生成字符串并返回
    @classmethod
    def popAndErr2str(cls, population, agent_set, num_parameters, flops, proxy_err, time):
        pop_str = []
        for id, particle in enumerate(population):
            _str = []
            _str.append('indi:%02d' % (id))
            _str.append('particle:%s' % (','.join(list(map(str, particle)))))
            _str.append('num_parameters:%d' % (num_parameters[id]))
            _str.append('FLOPs:%d' % (flops[id]))
            if proxy_err:
                _str.append('proxy_err:%.4f' % (proxy_err))
            _str.append('agent:%.4f' % (agent_set[id]))
            if time:
                _str.append('search time:%s' % (time))
            subparticle_length = cls.get_params('PSO', 'particle_length') // 3
            subParticles = [particle[0:subparticle_length], particle[subparticle_length:2 * subparticle_length],
                            particle[2 * subparticle_length:]]
            end_channel = 0
            for j, subParticle in enumerate(subParticles):
                valid_particle = cls.get_valid_particle(subParticle)
                if j > 0:
                    _str.append('Transition layer, stride=2, %d*%d MBConv3, in:%d, out:%d' % (
                        kernel_sizes[j + 1], kernel_sizes[j + 1], int(end_channel), int(end_channel) // 4))

                inchannels, outchannels, end_channel = cls.calc_in_out_channels(valid_particle, block_idx=j,
                                                                                end_channel=end_channel)
                _str.append('valid_subpar_%d:%s' % (j, ','.join(list(map(str, valid_particle)))))

                for number, dimen in enumerate(valid_particle):
                    in_channel = inchannels[number]
                    out_channel = outchannels[number]
                    _sub_str = []
                    conv_type = int(dimen)
                    # num_filters = int(valid_particle[i] - conv_type)*100
                    _sub_str.append('block%02d' % (j))
                    _sub_str.append('layer%02d' % (number))
                    _sub_str.append('stride=1')

                    if conv_type == 0:
                        _sub_str.append('Identity')
                    else:
                        _sub_str.append('%d*%d MBConv (expansion = %d), SE = %d' % (
                            kernel_sizes[conv_type], kernel_sizes[conv_type], expansion_rates[conv_type],
                            attent[conv_type]))

                    _sub_str.append('in:%d' % (in_channel))
                    _sub_str.append('out:%d' % (out_channel))

                    _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))

            _str.append('end_channel: %d' % (int(end_channel)))
            particle_str = '\n'.join(_str)
            pop_str.append(particle_str)
            pop_str.append('-' * 100)
        return '\n'.join(pop_str)

    # 不能超过给定的Identity 或者 Mbconv的配置表的范围 这段代码会创建一个新的列表dimens，其中包含particle列表中所有在0到12.99范围内的元素。
    @classmethod
    def get_valid_particle(cls, particle):
        return [dimen for dimen in particle if 0 <= dimen <= 12.99]

    # 计算有效粒子的输入输出以及最终的输出通道数
    @classmethod
    def calc_in_out_channels(cls, valid_particle, block_idx=None, end_channel=0):
        dataset = str(cls.__read_ini_file('NETWORK', 'dataset'))
        dataset_name = str(cls.__read_ini_file('NETWORK', 'name'))
        # 第一个密集块的卷积核需要初始化
        if block_idx == 0:
            if dataset == 'cifar10' or dataset == 'cifar100' or dataset_name == 'PathMNIST' or dataset_name == 'ChestMNIST' or dataset_name == 'DermaMNIST'or dataset_name == 'OCTMNIST'or dataset_name == 'PneumoniaMNIST'or dataset_name == 'RetinaMNIST'or dataset_name == 'BreastMNIST'or dataset_name == 'BloodMNIST'or dataset_name == 'TissueMNIST'or dataset_name == 'OrganAMNIST'or dataset_name == 'OrganCMNIST'or dataset_name == 'OrganSMNIST'or dataset_name == 'OrganMNIST3D'or dataset_name == 'NoduleMNIST3D'or dataset_name == 'AdrenalMNIST3D'or dataset_name == 'FractureMNIST3D'or dataset_name == 'VesselMNIST3D'or dataset_name == 'SynapseMNIST3D':
                node1_filter = 16
            elif dataset == 'imagenet' or dataset == 'LC25000' or dataset == 'BreakHis' or dataset == 'Colorectal' or dataset == 'road' or dataset == 'RSCD' or dataset == 'RTK':
                node1_filter = 32
        else:
            # 对于后续的密集块，第一个卷积层的卷积核数量为end_channel的1分之一
            node1_filter = end_channel // 1
        # 用于存储每个层的输入通道数（inchannels）和输出通道数（outchannels）
        inchannels = []
        outchannels = []
        # 如果开启了密集链接的模式
        if int(cls.__read_ini_file('NETWORK', 'dense')):
            for i in range(0, len(valid_particle)):
                dimen = valid_particle[i]
                if i == 0:
                    inchannel = node1_filter
                else:
                    # 对于后续层，输入通道数取决于前一层的输出通道数
                    # 如果前一层是Identity层，则输入通道数等于前一层的输出通道数
                    # 否则，输入通道数等于前一层的输入通道数和输出通道数的总和
                    if int(valid_particle[i - 1]) <= 0:
                        inchannel = outchannels[i - 1]
                    else:
                        inchannel = inchannels[i - 1] + outchannels[i - 1]
                inchannels.append(inchannel)

                # 改层为冻结层，输入通道等于输出通道
                if int(dimen) <= 0:
                    outchannels.append(inchannel)
                else:
                    # 计算输出通道数，涉及对dimen进行一些舍入和加法运算
                    outchannels.append(int(round(dimen - int(dimen), 2) * 100) + 1)
            # 如果最后一层是冻结层，将end_channel设置为最后一层的输出通道数
            if int(valid_particle[-1]) <= 0:
                end_channel = outchannels[-1]
            else:
                # 否则，输入通道数等于前一层的输入通道数和输出通道数的总和
                end_channel = inchannels[-1] + outchannels[-1]
        # 没有开启密集链接的模式 单链的方式
        else:
            for i in range(0, len(valid_particle)):
                dimen = valid_particle[i]
                if i == 0:
                    inchannel = node1_filter
                else:
                    inchannel = outchannels[i - 1]
                inchannels.append(inchannel)

                if int(dimen) <= 0:
                    outchannels.append(inchannel)
                else:
                    outchannels.append(int(round(dimen - int(dimen), 2) * 100) + 1)
            # 对于未开启dense模式，将end_channel设置为最后一层的输出通道数
            end_channel = outchannels[-1]
        # 返回计算得到的输入通道数列表、输出通道数列表和最终通道数 (将得到每个密集块里的Mbconv块的输入与输出通道数 做成对应的映射关系)
        return inchannels, outchannels, end_channel

    @classmethod
    def generate_forward_list(cls, particle, is_test):
        forward_list = []
        dataset_name = str(cls.__read_ini_file('NETWORK', 'name'))
        if dataset_name == 'OrganMNIST3D'or dataset_name == 'NoduleMNIST3D'or dataset_name == 'AdrenalMNIST3D'or dataset_name == 'FractureMNIST3D'or dataset_name == 'VesselMNIST3D'or dataset_name == 'SynapseMNIST3D':
            forward_list.append('out = self.Hswish(self.bn_begin1(self.conv_begin1(x.float())))')
        else:
            forward_list.append('out = self.Hswish(self.bn_begin1(self.conv_begin1(x)))')
        forward_list.append('out = self.evolved_block0(out)')
        forward_list.append('out = self.tranLayer0(out)')
        forward_list.append('out = self.evolved_block1(out)')
        forward_list.append('out = self.tranLayer1(out)')
        return forward_list

    @classmethod
    def read_template(cls, test_model=False):
        dataset = str(cls.__read_ini_file('NETWORK', 'dataset'))
        if test_model:
            _path = './template/' + dataset + '_test.py'
        else:
            _path = './template/' + dataset + '.py'
        part1 = []
        part2 = []
        part3 = []

        f = open(_path)
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part2))

        line = f.readline().rstrip()  # skip the comment '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part3))
        # part1 = 模板中generated_init的所有内容 part2 = 模板中generated_init之后与generate_forward之前的内容
        # part3 = 模板中generate_forward之后的所有内容
        return part1, part2, part3

    @classmethod
    def generate_pytorch_file(cls, particle, curr_gen, id, test_model=False):
        dataset = str(cls.__read_ini_file('NETWORK', 'dataset'))
        dataset_name = str(cls.__read_ini_file('NETWORK', 'name'))
        # 是否为最终训练和测试
        if curr_gen <= -1:
            is_test = True
        else:
            is_test = False

        act_funcs = ['relu', 'h_swish', 'h_swish', 'h_swish', 'h_swish']
        # 密集型连接需要 Drop Path是一种正则化技术，它被用来防止过拟合，特别是在训练深度神经网络时。
        if int(cls.__read_ini_file('NETWORK', 'dense')):
            drop_rates = [0.05, 0.1, 0.15, 0.20, 0.20]
        else:
            drop_rates = [0.0, 0.0, 0.0, 0.0, 0.0]
        # query convolution unit
        conv_list = []
        img_channel = cls.get_params('NETWORK', 'image_channel')

        # stem阶段
        if dataset_name == 'OrganMNIST3D'or dataset_name == 'NoduleMNIST3D'or dataset_name == 'AdrenalMNIST3D'or dataset_name == 'FractureMNIST3D'or dataset_name == 'VesselMNIST3D'or dataset_name == 'SynapseMNIST3D':
            conv_begin1 = 'self.conv_begin1 = nn.Conv3d(%d, %d, kernel_size=3, stride=2, padding=1, bias=False)' % (
            img_channel, 16)
            bn_begin1 = 'self.bn_begin1 = nn.BatchNorm3d(%d)' % (16)
        else:
            conv_begin1 = 'self.conv_begin1 = nn.Conv2d(%d, %d, kernel_size=3, stride=2, padding=1, bias=False)' % (
                img_channel, 16)
            bn_begin1 = 'self.bn_begin1 = nn.BatchNorm2d(%d)' % (16)
        conv_list.append(conv_begin1)
        conv_list.append(bn_begin1)

        # 计算每个块的子粒子的长度
        subparticle_length = cls.get_params('PSO', 'particle_length') // 3
        # 切割粒子为三部分，每部分是一个Block，每个block里面是密集链接网络（这些网络的基础组件都是mobilenet的MBconv组件构成）
        subParticles = [particle[0:subparticle_length], particle[subparticle_length:2 * subparticle_length],
                        particle[2 * subparticle_length:]]
        end_channel = 0
        # 每个block的信息进行编码生成pytorch文件
        for j, sub_particle in enumerate(subParticles):
            valid_particle = cls.get_valid_particle(sub_particle)
            # 层类型
            layer_types = []
            # 卷积核的大小
            kernelSizes = []
            # 扩大比率
            expansions = []
            # 注意力机制
            attentions = []
            # 生成长度为len(valid_particle) 值全为1的列表 步长列表
            strides = [1] * len(valid_particle)
            # 激活函数
            actFuncs = []
            # droprate的比率
            dropRates = []

            # 生成两个过渡层 只有大于1 ，2能生成过渡层 0不能生成过渡层
            if j > 0:
                # 生成过渡层 0 1
                stride_convname = 'self.tranLayer%d' % (j - 1)
                # 后续的密集块的输入都是前一个卷积核的输出通道数end_channel的1/2
                outC_strided = int(end_channel) // 1
                if dataset_name == 'OrganMNIST3D'or dataset_name == 'NoduleMNIST3D'or dataset_name == 'AdrenalMNIST3D'or dataset_name == 'FractureMNIST3D'or dataset_name == 'VesselMNIST3D'or dataset_name == 'SynapseMNIST3D':
                    strided_conv = '%s = MBConv3D(C_in=%d, C_out=%d, kernel_size=%d, expansion=3, stride=2, dilation=1, act_func="%s", attention=0, drop_connect_rate=0, dense=%d)' % (
                    stride_convname, int(end_channel), outC_strided, kernel_sizes[j + 1], act_funcs[j],
                    int(cls.__read_ini_file('NETWORK', 'dense')))
                else:    
                    strided_conv = '%s = MBConv(C_in=%d, C_out=%d, kernel_size=%d, expansion=3, stride=2, dilation=1, act_func="%s", attention=0, drop_connect_rate=0, dense=%d)' % (
                        stride_convname, int(end_channel), outC_strided, kernel_sizes[j + 1], act_funcs[j],
                        int(cls.__read_ini_file('NETWORK', 'dense')))
                conv_list.append(strided_conv)
            # 拿到当前密集块的输入、输出、和最后的输出通道数 end_channel为过渡层使用
            in_channels, out_channels, end_channel = cls.calc_in_out_channels(valid_particle, block_idx=j,
                                                                              end_channel=end_channel)
            # 根据每个有效子粒子进行解码获取对应密集块里的Mbconv的配置信息
            for i, dimen in enumerate(valid_particle):
                # 获取该层的类型，Mbconv或者Identity
                layer_type = int(dimen)
                layer_types.append(layer_type)
                # 根据map的对应信息获取该层的卷积核大小 注意力机制 扩展比率 激活函数 drop比率
                kernelSizes.append(kernel_sizes[layer_type])
                attentions.append(attent[layer_type])
                expansions.append(expansion_rates[layer_type])
                actFuncs.append(act_funcs[j])
                dropRates.append(drop_rates[j])
            # 生成密集块的pytorch文件
            conv_name = 'self.evolved_block%d' % j
            if dataset_name == 'OrganMNIST3D'or dataset_name == 'NoduleMNIST3D'or dataset_name == 'AdrenalMNIST3D'or dataset_name == 'FractureMNIST3D'or dataset_name == 'VesselMNIST3D'or dataset_name == 'SynapseMNIST3D':
                conv = '%s = DenseBlock3D(layer_types="%s", in_channels="%s", out_channels="%s", kernel_sizes="%s", expansions="%s", strides="%s", act_funcs="%s", attentions="%s", drop_connect_rates="%s", dense="%d")' % (
                conv_name, " ".join(str(i) for i in layer_types), " ".join(str(i) for i in in_channels),
                " ".join(str(i) for i in out_channels),
                " ".join(str(i) for i in kernelSizes), " ".join(str(i) for i in expansions),
                " ".join(str(i) for i in strides), " ".join(str(i) for i in actFuncs),
                " ".join(str(i) for i in attentions), " ".join(str(i) for i in dropRates),
                int(cls.__read_ini_file('NETWORK', 'dense')))
            else:
                conv = '%s = DenseBlock(layer_types="%s", in_channels="%s", out_channels="%s", kernel_sizes="%s", expansions="%s", strides="%s", act_funcs="%s", attentions="%s", drop_connect_rates="%s", dense="%d")' % (
                    conv_name, " ".join(str(i) for i in layer_types), " ".join(str(i) for i in in_channels),
                    " ".join(str(i) for i in out_channels),
                    " ".join(str(i) for i in kernelSizes), " ".join(str(i) for i in expansions),
                    " ".join(str(i) for i in strides), " ".join(str(i) for i in actFuncs),
                    " ".join(str(i) for i in attentions), " ".join(str(i) for i in dropRates),
                    int(cls.__read_ini_file('NETWORK', 'dense')))
            conv_list.append(conv)

        # query fully-connect layer, because a global avg_pooling layer is added before the fc layer, so the input size of the fc layer is equal to out channel
        if dataset == 'cifar10':
            factor = 1
            num_class = 10
        elif dataset == 'LC25000':
            factor = 16
            num_class = 5
        elif dataset == 'BreakHis':
            factor = 16
            num_class = 8
        elif dataset == 'Colorectal':
            factor = 16
            num_class = 8
        elif dataset == 'road':
            factor = 16
            num_class = 8
        elif dataset == 'RSCD':
            factor = 16
            num_class = 13
        elif dataset == 'RTK':
            factor = 16
            num_class = 8
        
        elif dataset_name == 'PathMNIST':
            factor = 1
            num_class = 9
        elif dataset_name == 'ChestMNIST':
            factor = 1
            num_class = 14
        elif dataset_name == 'DermaMNIST':
            factor = 1
            num_class = 7
        elif dataset_name == 'OCTMNIST':
            factor = 1
            num_class = 4
        elif dataset_name == 'PneumoniaMNIST':
            factor = 1
            num_class = 2
        elif dataset_name == 'RetinaMNIST':
            factor = 1
            num_class = 5
        elif dataset_name == 'BreastMNIST':
            factor = 1
            num_class = 2
        elif dataset_name == 'BloodMNIST':
            factor = 1
            num_class = 8
        elif dataset_name == 'TissueMNIST':
            factor = 1
            num_class = 8
        elif dataset_name == 'OrganAMNIST':
            factor = 1
            num_class = 11
        elif dataset_name == 'OrganCMNIST':
            factor = 1
            num_class = 11
        elif dataset_name == 'OrganSMNIST':
            factor = 1
            num_class = 11
        elif dataset_name == 'OrganMNIST3D':
            factor = 1
            num_class = 11
        elif dataset_name == 'NoduleMNIST3D':
            factor = 1
            num_class = 2
        elif dataset_name == 'AdrenalMNIST3D':
            factor = 1
            num_class = 2
        elif dataset_name == 'FractureMNIST3D':
            factor = 1
            num_class = 3
        elif dataset_name == 'VesselMNIST3D':
            factor = 1
            num_class = 2
        elif dataset_name == 'SynapseMNIST3D':
            factor = 1
            num_class = 2

        fc_node = factor * end_channel

        # add an 1*1 conv to the end of the conv_list, before global avg pooling  2021.03.05
        if dataset_name == 'OrganMNIST3D'or dataset_name == 'NoduleMNIST3D'or dataset_name == 'AdrenalMNIST3D'or dataset_name == 'FractureMNIST3D'or dataset_name == 'VesselMNIST3D'or dataset_name == 'SynapseMNIST3D':
            conv_end1 = 'self.conv_end1 = nn.Conv3d(%d, %d, kernel_size=1, stride=1, bias=False)' % (end_channel, fc_node)
            conv_list.append(conv_end1)
            bn_end1 = 'self.bn_end1 = nn.BatchNorm3d(%d)' % (fc_node)
            conv_list.append(bn_end1)
        else:
            conv_end1 = 'self.conv_end1 = nn.Conv2d(%d, %d, kernel_size=1, stride=1, bias=False)' % (end_channel, fc_node)
            conv_list.append(conv_end1)
            bn_end1 = 'self.bn_end1 = nn.BatchNorm2d(%d)' % (fc_node)
            conv_list.append(bn_end1)
        fully_layer_name4 = 'self.kan_linear = KANLinear(%d, %d)' % (fc_node, num_class)

        # generate the forward part
        forward_list = cls.generate_forward_list(particle, is_test)

        part1, part2, part3 = cls.read_template(test_model)
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\n        %s' % ('#conv unit'))
        for s in conv_list:
            _str.append('        %s' % (s))
        _str.append('\n        %s' % ('#linear unit'))
        _str.append('        %s' % (fully_layer_name4))

        _str.extend(part2)
        for s in forward_list:
            _str.append('        %s' % (s))
        _str.extend(part3)
        # print('\n'.join(_str))
        file_path = './scripts/particle%02d_%02d.py' % (curr_gen, id)
        script_file_handler = open(file_path, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
        file_name = 'particle%02d_%02d' % (curr_gen, id)
        return file_name


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("MPSZ-NAS")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warning(_str)


class GPUTools(object):
    @classmethod
    def _get_equipped_gpu_ids_and_used_gpu_info(cls):
        p = Popen('nvidia-smi', stdout=PIPE)
        output_info = p.stdout.read().decode('UTF-8')
        lines = output_info.split(os.linesep)
        equipped_gpu_ids = []
        for line_info in lines:
            if not line_info.startswith(' '):
                if 'GeForce' in line_info or 'Quadro' in line_info or 'Tesla' in line_info or 'RTX' in line_info or 'A40' in line_info:
                    equipped_gpu_ids.append(line_info.strip().split(' ', 4)[3])
            else:
                break

        gpu_info_list = []
        for line_no in range(len(lines) - 3, -1, -1):
            if lines[line_no].startswith('|==='):
                break
            else:
                gpu_info_list.append(lines[line_no][1:-1].strip())

        return equipped_gpu_ids, gpu_info_list

    @classmethod
    def get_available_gpu_ids(cls):
        equipped_gpu_ids, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))

        unused_gpu_ids = []
        for id_ in equipped_gpu_ids:
            if id_ not in used_gpu_ids:
                unused_gpu_ids.append(id_)
        return unused_gpu_ids

    @classmethod
    def detect_available_gpu_id(cls):
        unused_gpu_ids = cls.get_available_gpu_ids()
        if len(unused_gpu_ids) == 0:
            Log.info('GPU_QUERY-No available GPU')
            return None
        else:
            num = 0
            if str(num) in unused_gpu_ids:
                # Log.info('GPU_QUERY-No available GPU')
                Log.info('GPU_QUERY-Available GPUs are: [%s], choose GPU#%d to use' % (','.join(unused_gpu_ids), num))
                return num
            else:
                Log.info(
                    'GPU_QUERY-Available GPUs are: [%s], but GPU#%d is unavailable' % (','.join(unused_gpu_ids), num))
                return None

    @classmethod
    def all_gpu_available(cls):
        _, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))
        if len(used_gpu_ids) == 0:
            Log.info('GPU_QUERY-None of the GPU is occupied')
            return True
        else:
            Log.info('GPU_QUERY- GPUs [%s] are occupying' % (','.join(used_gpu_ids)))
            return False
