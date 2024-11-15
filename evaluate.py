# !/usr/bin/python
# -*- coding: utf-8 -*-

from utils import Utils, Log, GPUTools
from multiprocessing import Process
import importlib
import sys, os, time
import numpy as np
import copy
from asyncio.tasks import sleep


def decode(particle, curr_gen, id):
    pytorch_filename = Utils.generate_pytorch_file(particle, curr_gen, id)

    return pytorch_filename


def check_all_finished(filenames, curr_gen):
    filenames_ = copy.deepcopy(filenames)
    output_file = './populations/err_%02d.txt' % (curr_gen)
    if os.path.exists(output_file):
        f = open(output_file, 'r')
        for line in f:
            if len(line.strip()) > 0:
                line = line.strip().split('=')
                if line[0] in filenames_:
                    filenames_.remove(line[0])
        f.close()
        if filenames_:
            return False
        else:
            return True
    else:
        return False


def fitnessEvaluate(filenames, curr_gen, is_test, population, batch_size_set=None, weight_decay_set=None):
    # 初始化：创建一个数组以存储每个个体的错误、参数和FLOPS值。
    zen_score_params_flops_set = np.zeros(shape=(3, len(filenames)))
    has_evaluated_offspring = False
    p = None
    # 遍历每一个架构进行训练和评估
    for i, file_name in enumerate(filenames):
        # 开始之后评估个体就会被设置成True
        has_evaluated_offspring = True
        # time.sleep(40)
        # 如果有现有进程，等待其完成
        if p:
            p.join()
        gpu_id = GPUTools.detect_available_gpu_id()
        while gpu_id is None:
            time.sleep(60)
            gpu_id = GPUTools.detect_available_gpu_id()
        # 开始训练当前个体
        if gpu_id is not None:
            Log.info('Begin to train %s' % (file_name))
            module_name = 'scripts.%s' % (file_name)
            # 检查模块是否已加载，如果是，则删除它，然后加载当前模块，确保模块不被重复加载
            if module_name in sys.modules.keys():
                Log.info('Module:%s has been loaded, delete it' % (module_name))
                del sys.modules[module_name]
                _module = importlib.import_module('.', module_name)
            else:
                _module = importlib.import_module('.', module_name)
            # 从模块导入 'RunModel' 类
            _class = getattr(_module, 'RunModel')
            cls_obj = _class()
            # 创建一个进程，运行 'RunModel' 的 'do_work' 方法，种群中的每个粒子构成的架构都会进行训练评估
            if batch_size_set:
                # 最终测试的进程
                p = Process(target=cls_obj.do_work, args=(
                    '%d' % (gpu_id), curr_gen, file_name, is_test, population[i], batch_size_set[i],
                    weight_decay_set[i]))
            else:
                # 搜索阶段适应性评估的进程
                p = Process(target=cls_obj.do_work, args=('%d' % (gpu_id), curr_gen, file_name, is_test, population[i]))
            p.start()

    # 等待最后一个进程完成
    p.join()
    time.sleep(10)

    # 如果已评估个体
    if has_evaluated_offspring:
        # 定义用于读取健身值的文件名
        file_names = ['./populations/agent_%02d.txt' % (curr_gen), './populations/params_%02d.txt' % (curr_gen),
                      './populations/flops_%02d.txt' % (curr_gen)]
        # 初始化适应度值映射以存储从文件中读取的值
        fitness_maps = [{}, {}, {}]
        # 从文件中读取健身值并填充 fitness_maps
        for j, file_name in enumerate(file_names):
            assert os.path.exists(file_name) == True
            f = open(file_name, 'r')

            for line in f:
                if len(line.strip()) > 0:
                    line = line.strip().split('=')
                    fitness_maps[j][line[0]] = float(line[1])
            f.close()

            # 使用从文件中读取的健身值更新 err_params_flops_set
            for i in range(len(zen_score_params_flops_set[0])):
                if filenames[i] not in fitness_maps[j]:
                    Log.warn(
                        'The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 60 seconds' % (
                            filenames[i], file_name))
                    sleep(120)
                zen_score_params_flops_set[j][i] = fitness_maps[j][filenames[i]]

    else:
        Log.info('None offspring has been evaluated')
    # 返回zen_score 参数量 浮点数 均为列集合
    return list(zen_score_params_flops_set[0]), list(zen_score_params_flops_set[1]), list(zen_score_params_flops_set[2])
