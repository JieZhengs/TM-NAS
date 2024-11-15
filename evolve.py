# !/usr/bin/python
# -*- coding: utf-8 -*-
import configparser
import random
from scipy.stats import beta
import numpy as np
import math
from utils import Log

def __read_ini_file(section, key):
    config = configparser.ConfigParser()
    config.read('global.ini')
    return config.get(section, key)


def random_excluding_specific_values():
    # 生成一个 [0, 1) 范围内的随机数
    random_number = random.uniform(0, 1)

    # 检查并排除特定的值
    while random_number in (0, 0.25, 0.75, 1):
        random_number = random.uniform(0, 1)

    return random_number


def logistic_map(w_max, w_min, T_max, iterations, sigma, b1, b2):
    # 初始化 r 的初始值
    r = random_excluding_specific_values()
    for t in range(0, T_max):
        # 应用 Logistic 映射更新 r
        r = 4 * r * (1 - r)
        # 确保 r 在 [0, 1] 范围内
        r = max(0, min(1, r))
        # 计算惯性权重 w
        w = r * w_min + (w_max - w_min) * t / T_max
        w = (1 - w) + sigma * beta.rvs(b1, b2)
        if t == iterations:
            return w
    return w_min


def dynamic_learning_factors(k, max_iter, c_max, c_min):
    # 计算学习因子 c1 和 c2
    c1 = c_max - (c_max - c_min) * math.sin(math.pi * k / (2 * max_iter))
    c2 = c_min + (c_max - c_min) * math.sin(math.pi * k / (2 * max_iter))
    return c1, c2


def adapt_inertia_weight(weight_set, c1_set, c2_set, best_agent_set, previous_agent_set, best_num_parameters,
                         best_flops,
                         gbest_agent, curr_gen):
    sigma = float(__read_ini_file('SEARCH', 'sigma'))
    b1 = int(__read_ini_file('SEARCH', 'b1'))
    b2 = int(__read_ini_file('SEARCH', 'b2'))
    TMAX = int(__read_ini_file('PSO', 'num_iteration'))

    w_min = float(__read_ini_file('SEARCH', 'w_min'))
    w_max = float(__read_ini_file('SEARCH', 'w_max'))
    c_min = float(__read_ini_file('SEARCH', 'c_min'))
    c_max = float(__read_ini_file('SEARCH', 'c_max'))
    Log.info('动态自适应更改pso权重中....')

    # 方法一
    for i in range(0, len(weight_set)):
        weight_set[i] = logistic_map(w_max, w_min, TMAX, curr_gen, sigma, b1, b2)
        c1_set[i], c2_set[i] = dynamic_learning_factors(curr_gen, TMAX, c_max, c_min)
    
    # 方法二
    # for i in range(0, len(weight_set)):
    #     # 如果适应度下降，则增加惯性权重
    #     # 如果适应度下降，则降低来自自身的经验，更趋近于全局经验
    #     c1_set[i] *= pow(Tp / best_num_parameters[i], wp[int(bool(best_num_parameters[i] > Tp))]) * pow(
    #         Tf / best_flops[i],
    #         wf[int(bool(best_flops[i] > Tf))])
    #     c2_set[i] *= pow((1 - gbest_err) / (1 - err_set[i]), wa[int(bool((1 - err_set[i]) > (1 - gbest_err)))])
    #     weight_set[i] *= pow((1 - gbest_err) / (1 - err_set[i]), wa[int(bool((1 - err_set[i]) > (1 - gbest_err)))])

    # 方法三
    # for j in range(0, len(best_agent_set)):
    #     if best_agent_set[j] < previous_agent_set[j]:
    #         for i in range(0, len(weight_set)):
    #             # 如果适应度下降，则增加惯性权重
    #             # 如果适应度下降，则降低来自自身的经验，更趋近于全局经验
    #             c1_set[i] *= 1.1
    #             c2_set[i] *= 1.1
    #             weight_set[i] *= 1.1
    #     elif best_agent_set[j] > previous_agent_set[j]:
    #         # 如果适应度上升，则减小惯性权重
    #         # 如果适应度上升，则增加来自自身的经验，适当降低来自全局经验，探索更多样性的可能
    #         for i in range(0, len(weight_set)):
    #             c1_set[i] *= 0.9
    #             c2_set[i] *= 0.9
    #             weight_set[i] *= 0.9

    return weight_set, c1_set, c2_set


def calculate_pi(fitness, fitness_sum, N):
    # 计算 pi 的分子
    numerator = np.exp(fitness)

    # 计算 pi 的分母
    denominator = np.exp(fitness_sum / N)

    # 计算 pi
    pi = numerator / denominator
    return pi


def aconpso(weight_set, c1_set, c2_set, particle, gbest, pbest, velocity, params,
            previous_agent_set,
            best_agent, best_agent_set, best_num_parameters, gbest_agent, best_flops, curr_gen):
    # print('gbest' + str(gbest))
    """
    pso for architecture evolution
    fixed-length PSO, use standard formula, but add a strided layer number constraint
    """
    particle_length = params['particle_length']
    max_output_channel = params['max_output_channel']
    N = int(__read_ini_file('PSO', 'pop_size'))
    cur_len = len(particle)

    # 1.velocity calculation
    # weight, c1, c2 = 0.7298, 1.49618, 1.49618

    # 根据当前的错误率自适应惯性权重
    weight_set, c1_set, c2_set = adapt_inertia_weight(weight_set, c1_set, c2_set, best_agent_set,
                                                      previous_agent_set,
                                                      best_num_parameters, best_flops, gbest_agent, curr_gen)
    r1 = np.random.random(cur_len)
    r2 = np.random.random(cur_len)
    delta = random.choice([-1, 0])
    m = random.uniform(0.4, 0.9)
    beta = random.random()
    # 粒子的下一个速度动量的计算公式 动量 = 自身惯性权重 + 个体最佳解向量 + 全局最佳解向量
    pi = calculate_pi(best_agent, sum(best_agent_set), N)
    new_velocity = np.zeros(len(weight_set))
    new_particle = np.zeros(len(weight_set))
    # print('pi' + str(len(pi)))
    for i in range(0, len(weight_set)):
        if pi < 0.6:
            # Log.info('当前个人最佳解与全局平均解比重小于0.6，直线形式，加大探索力度帮助最佳架构搜索')
            new_velocity[i] = velocity[i] * weight_set[i] + math.pow(-1, delta) * c1_set[i] * r1[i] * (
                    (pbest[i] + gbest[i]) / 2 - m * particle[i]) + math.pow(-1, delta) * c2_set[i] * r2[i] * (
                                      (pbest[i] - gbest[i]) / 2 - m * particle[i])
            new_particle[i] = weight_set[i] * particle[i] + (1 - weight_set[i]) * new_velocity[i]
        else:
            # Log.info('当前个人最佳解与全局平均解比重大于等于0.6，盘旋形式，减少探索力度帮助收敛')
            new_velocity[i] = velocity[i] * weight_set[i] + math.pow(-1, delta) * c1_set[i] * r1[i] * (
                    (pbest[i] + gbest[i]) / 2 - math.cos(2 * np.pi * beta) * particle[i]) + math.pow(-1, delta) * \
                              c2_set[
                                  i] * r2[i] * (
                                      (pbest[i] - gbest[i]) / 2 - math.cos(2 * np.pi * beta) * particle[i])
            new_particle[i] = particle[i] + new_velocity[i]
    # new_velocity = np.asarray(velocity) * np.asarray(weight_set) + np.asarray(c1_set) * r1 * (
    #         np.asarray(pbest) - np.asarray(particle)) + np.asarray(c2_set) * r2 * (
    #                        np.asarray(gbest) - np.asarray(particle))

    # 2.particle updating
    # new_particle = list(particle + new_velocity)
    new_particle = [round(par, 2) for par in new_particle]  # particle里面的数必须为两位小数
    new_particle = [abs(par) for par in new_particle]
    # new_velocity = list(new_velocity)
    Log.info('new_particle' + str(new_particle))

    # 3.adjust the value according to some constraints
    # 保证子粒子满足限制条件且可用
    subparticle_length = particle_length // 3
    subParticles = [new_particle[0:subparticle_length], new_particle[subparticle_length:2 * subparticle_length],
                    new_particle[2 * subparticle_length:]]

    for j, subParticle in enumerate(subParticles):
        # 获取有效粒子
        valid_particle = [dimen for dimen in subParticle if 0 <= dimen <= 12.99]
        # condition 1：the number of valid layer (non-strided or strided layer, not identity) must >0
        if len(valid_particle) == 0:
            # if the updated particle has no valid value, let the first dimension value to 0.03 (3*3 DW-sep conv, no.filter=3)
            new_particle[j * subparticle_length] = 0.00

    # 4.outlier handling - maintain the particle and velocity within their valid ranges
    # 保持该粒子的范围在有效值之内
    updated_particle1 = []
    for k, par in enumerate(new_particle):
        if (0.00 <= par <= 12.99):
            updated_particle1.append(par)
        elif par > 12.99:
            updated_particle1.append(12.99)
        else:
            updated_particle1.append(0.00)

    updated_particle = []
    for k, par in enumerate(updated_particle1):
        if int(round(par - int(par), 2) * 100) + 1 > max_output_channel:
            updated_particle.append(round(int(par) + float(max_output_channel - 1) / 100, 2))
        else:
            updated_particle.append(par)

    return updated_particle, new_velocity, weight_set, c1_set, c2_set
