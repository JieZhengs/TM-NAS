# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from utils import Utils, Log, GPUTools
from population import initialize_population
from evaluate import decode, fitnessEvaluate
from evolve import aconpso
import copy, os, time
import configparser


def create_directory():
    dirs = ['./log', './populations', './scripts', './trained_models']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


# 评估阶段的适应度评估
def fitness_evaluate(population, curr_gen):
    filenames = []
    population_downscale = []
    for i, particle in enumerate(population):
        # 架构降尺度
        # 如果 dimen 的小数部分大于等于 0.02，则将 dimen 转换为整数部分加上小数部分的一半，但要减去 0.01。这通过 int(dimen) + round((dimen - int(dimen) + 0.01) / 2 - 0.01, 2) 实现。
        # 如果 dimen 的小数部分小于 0.02，则将 dimen 转换为其整数部分。这通过 round(dimen // 1 + 0.00, 2) 实现，不要小数部分。
        particle_downscale = [int(dimen) + round((dimen - int(dimen) + 0.01) / 2 - 0.01, 2) if round(dimen - int(dimen),
                                                                                                     2) >= 0.02 else round(
            dimen // 1 + 0.00, 2) for dimen in particle]
        # particle = copy.deepcopy(particle)
        # 种群中每个粒子生成对应的pytorch文件
        filename = decode(particle_downscale, curr_gen, i)
        filenames.append(filename)
        population_downscale.append(particle_downscale)

    agent, num_parameters, flops = fitnessEvaluate(filenames, curr_gen, is_test=False,
                                                       population=population_downscale)
    return agent, num_parameters, flops


# 进化粒子更具个人的惯性权重 全局最佳解 个体最佳解进行粒子更新
def evolve(weight_set, c1_set, c2_set, population, gbest_individual, pbest_individuals, velocity_set,
           params,
           previous_agent_set, best_agent_set, best_num_parameters, gbest_agent, best_flops,
           curr_gen):
    offspring = []
    new_velocity_set = []
    new_weight_set = []
    new_c1_set = []
    new_c2_set = []
    for i, particle in enumerate(population):
        new_particle, new_velocity, new_weight, new_c1, new_c2 = aconpso(weight_set[i], c1_set[i],
                                                                         c2_set[i], particle,
                                                                         gbest_individual,
                                                                         pbest_individuals[i],
                                                                         velocity_set[i], params,
                                                                         previous_agent_set,
                                                                         best_agent_set[i], best_agent_set,
                                                                         best_num_parameters,
                                                                         gbest_agent,
                                                                         best_flops,
                                                                         curr_gen)
        offspring.append(new_particle)
        new_velocity_set.append(new_velocity)
        new_weight_set.append(new_weight)
        new_c1_set.append(new_c1)
        new_c2_set.append(new_c2)
    return offspring, new_velocity_set, new_weight_set, new_c1_set, new_c2_set


# 更新个体最佳解和全局最佳解的粒子信息
def update_best_particle(population, best_agent_set, num_parameters, flops, gbest, pbest):
    # 生成当前代种群个体最佳解的适应度值集合
    fitnessSet = [
        best_agent_set[i] * pow(num_parameters[i] / Tp, wp[int(bool(num_parameters[i] > Tp))]) * pow(flops[i] / Tf,
                                                                                                         wf[
                                                                                                             int(bool(
                                                                                                                 flops[
                                                                                                                     i] > Tf))])
        for i in range(len(population))]
    # 使用个体最佳解 初始话全局最佳解
    if not pbest:
        pbest_individuals = copy.deepcopy(population)
        pbest_agent_Set = copy.deepcopy(best_agent_set)
        pbest_params = copy.deepcopy(num_parameters)
        pbest_flops = copy.deepcopy(flops)
        gbest_individual, gbest_agent, gbest_params, gbest_flops, gbest_fitness = getGbest(
            [pbest_individuals, pbest_agent_Set, pbest_params, pbest_flops])
    # 存在个体最佳解求更新全局最佳解
    else:
        gbest_individual, gbest_agent, gbest_params, gbest_flops = gbest
        pbest_individuals, pbest_agent_Set, pbest_params, pbest_flops = pbest

        # 当前代种群的个体最佳解集合
        pbest_fitnessSet = [
            pbest_agent_Set[i] * pow(pbest_params[i] / Tp, wp[int(bool(pbest_params[i] > Tp))]) * pow(
                pbest_flops[i] / Tf, wf[int(bool(pbest_flops[i] > Tf))]) for i in range(len(pbest_individuals))]

        # 计算当前的全局最佳解的适应度值
        gbest_fitness = gbest_agent * pow(gbest_params / Tp, wp[int(bool(gbest_params > Tp))]) * pow(
            gbest_flops / Tf, wf[int(bool(gbest_flops > Tf))])

        # 更新个体最佳解的适应度集合
        for i, fitness in enumerate(fitnessSet):
            if fitness > pbest_fitnessSet[i]:
                pbest_individuals[i] = copy.deepcopy(population[i])
                pbest_agent_Set[i] = copy.deepcopy(best_agent_set[i])
                pbest_params[i] = copy.deepcopy(num_parameters[i])
                pbest_flops[i] = copy.deepcopy(flops[i])
            # 如果当前的个体最佳解大于全局最佳解 则更新当前个体最佳解为全局最佳解
            if fitness > gbest_fitness:
                gbest_fitness = copy.deepcopy(fitness)
                gbest_individual = copy.deepcopy(population[i])
                gbest_agent = copy.deepcopy(best_agent_set[i])
                gbest_params = copy.deepcopy(num_parameters[i])
                gbest_flops = copy.deepcopy(flops[i])

    return [gbest_individual, gbest_agent, gbest_params, gbest_flops], [pbest_individuals, pbest_agent_Set,
                                                                            pbest_params,
                                                                            pbest_flops]


# 获取全局最佳解
def getGbest(pbest):
    pbest_individuals, pbest_agent_Set, pbest_params, pbest_flops = pbest
    gbest_agent = float('-inf')
    gbest_params = 10e6
    gbest_flops = 10e9
    gbest = None

    # 初始化全局最佳解
    gbest_fitness = gbest_agent * pow(gbest_params / Tp, wp[int(bool(gbest_params > Tp))]) * pow(gbest_flops / Tf,
                                                                                                     wf[int(bool(
                                                                                                         gbest_flops > Tf))])

    # 遍历粒子的每个个体最佳解得到一个个体最佳解的集合
    pbest_fitnessSet = [pbest_agent_Set[i] * pow(pbest_params[i] / Tp, wp[int(bool(pbest_params[i] > Tp))]) * pow(
        pbest_flops[i] / Tf, wf[int(bool(pbest_flops[i] > Tf))]) for i in range(len(pbest_individuals))]

    # 遍历个体最佳解的集合更新全局最佳解
    for i, indi in enumerate(pbest_individuals):
        # 如果当前个体最佳解大于全局最佳解则更新当前的全局最佳解为当前的个体最佳解
        if pbest_fitnessSet[i] > gbest_fitness:
            gbest = copy.deepcopy(indi)
            gbest_agent = copy.deepcopy(pbest_agent_Set[i])
            gbest_params = copy.deepcopy(pbest_params[i])
            gbest_flops = copy.deepcopy(pbest_flops[i])
            gbest_fitness = copy.deepcopy(pbest_fitnessSet[i])
    return gbest, gbest_agent, gbest_params, gbest_flops, gbest_fitness


# 最终的适应度训练和测试
def fitness_test(final_individual):
    final_individual = copy.deepcopy(final_individual)
    filename = Utils.generate_pytorch_file(final_individual, -1, -1)
    agents, num_parameters, flops = fitnessEvaluate([filename], -1, True, [final_individual], [batch_size],
                                                        [weight_decay])
    return agents[0], num_parameters[0], flops[0]


def calculate_fitness(agent, num_parameters, flops):
    agent_set_mean = np.mean(agent)
    num_parameters_mean = np.mean(num_parameters)
    flops_mean = np.mean(flops)
    weighted_avg = agent_set_mean * 0.7 - num_parameters_mean / 1000000 * 0.15 - flops_mean / 1000000 * 0.15
    return weighted_avg


def evolveCNN(params):
    # 第一代的种群适应度评估，初始化个体最佳解 全局最佳解
    gen_no = 0
    Log.info('Initialize...')
    start = time.time()
    Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness（第一代初始化种群的的适应度评估）' % (gen_no))
    weight_avg = float('-inf')  # 初始化为负无穷
    best_population = None
    best_agent_set = None
    best_num_parameters = None
    best_flops = None
    for i in range(0, start_init_gen):
        population = initialize_population(params)
        Log.info('第 %d 个population：' % i + ':' + str(population))
        agent, num_parameters, flops = fitness_evaluate(population, -(i + 1))
        Log.info('第 %d 个种群的平均agent：' % i + ':' + str(np.mean(agent)))
        current_weight_avg = calculate_fitness(agent, num_parameters,
                                               flops)
        if current_weight_avg > weight_avg:
            weight_avg = current_weight_avg
            best_agent_set = agent
            best_num_parameters = num_parameters
            best_flops = flops
            best_population = population

    Log.info('初始化得到的最佳polulation：' + str(best_population))
    Log.info('初始化第一代运行的最佳架构agent均值' + str(np.mean(best_agent_set)))
    Log.info('EVOLVE[%d-gen]-当前代粒子群更新初始化的最佳加权适应度[ %d ],最佳初始化agent[ %.4f ]' % (
        gen_no, weight_avg, np.mean(best_agent_set)))
    previous_agent_set = best_agent_set
    Log.info('EVOLVE[%d-gen]-Finish the evaluation（第一代初始化种群的适应度评估完成）' % (gen_no))
    # update gbest and pbest, each individual contains two vectors, vector_archit and vector_conn
    [gbest_individual, gbest_agent, gbest_params, gbest_flops], [pbest_individuals, pbest_agent_Set,
                                                                     pbest_params,
                                                                     pbest_flops] = update_best_particle(
        best_population,
        best_agent_set,
        best_num_parameters,
        best_flops,
        gbest=None,
        pbest=None)
    Log.info('第一代的gbest_agent' + str(gbest_agent))
    Log.info('第一代的pbest_agent_Set' + str(np.mean(pbest_agent_Set)))
    Log.info('EVOLVE[%d-gen]-Finish the updating（初始化第一代的个体最佳解、全局最佳解完成）' % (gen_no))

    # 第一代的种群根据当前的种群 错误率 参数 浮点运算数 代数进行结果保存
    Utils.save_population_and_agent('populations', best_population, best_agent_set, best_num_parameters,
                                        best_flops, gen_no)
    Utils.save_population_and_agent('pbest', pbest_individuals, pbest_agent_Set, pbest_params, pbest_flops,
                                        gen_no)
    Utils.save_population_and_agent('gbest', [gbest_individual], [gbest_agent], [gbest_params], [gbest_flops],
                                        gen_no)

    gen_no += 1
    velocity_set = []
    weight_set = []
    c1_set = []
    c2_set = []
    # 初始化种群中每个粒子的速度动量都为0.01
    for ii in range(len(best_population)):
        # 生成速度向量集合长度为len(populations[ii])值全为0.01的集合
        velocity_set.append([0.01] * len(best_population[ii]))
        weight_set.append([weight] * len(best_population[ii]))
        c1_set.append([c1] * len(best_population[ii]))
        c2_set.append([c2] * len(best_population[ii]))

    # 权重查看
    # Log.info('运行前pso惯性权重：' + str(weight_set))
    # Log.info('运行前c1个人因子权重：' + str(c1_set))
    # Log.info('运行前c2社会因子权重：' + str(c2_set))

    # 第二代及之后的所有代的种群都要进行num_iteration - 1次的迭代更新
    for curr_gen in range(gen_no, params['num_iteration']):
        params['gen_no'] = curr_gen
        Log.info('EVOLVE[%d-gen]-Begin pso evolution（当前代粒子群更新）' % (curr_gen))

        Log.info('前一代架构运行的agent' + str(np.mean(previous_agent_set)))
        Log.info('后一代及以后的架构运行的agent' + str(np.mean(best_agent_set)))
        # Log.info('进化前 %d 代pso惯性权重：' % (curr_gen) + str(weight_set))
        # Log.info('进化前 %d 代个人因子c1：' % (curr_gen) + str(c1_set))
        # Log.info('进化前 %d 代社会因子c2：' % (curr_gen) + str(c2_set))

        best_population, velocity_set, weight_set, c1_set, c2_set = evolve(weight_set, c1_set, c2_set,
                                                                           best_population,
                                                                           gbest_individual,
                                                                           pbest_individuals,
                                                                           velocity_set,
                                                                           params,
                                                                           previous_agent_set,
                                                                           best_agent_set, best_num_parameters,
                                                                           gbest_agent,
                                                                           best_flops,
                                                                           curr_gen)
        # Log.info('进化后 %d 代pso惯性权重：' % (curr_gen) + str(weight_set))
        # Log.info('进化后 %d 代个人因子c1：' % (curr_gen) + str(c1_set))
        # Log.info('进化后 %d 代社会因子c2：' % (curr_gen) + str(c2_set))

        Log.info('EVOLVE[%d-gen]-Finish pso evolution（当前代粒子群更新完成）' % (curr_gen))
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness（当前代粒子适应度评估）' % (curr_gen))
        previous_agent_set = best_agent_set
        # 适应度评估
        best_agent_set, best_num_parameters, best_flops = fitness_evaluate(best_population, curr_gen)
        Log.info('EVOLVE[%d-gen]-Finish the evaluation（当前代粒子适应度评估完成）' % (curr_gen))

        [gbest_individual, gbest_agent, gbest_params, gbest_flops], [pbest_individuals, pbest_agent_Set,
                                                                         pbest_params,
                                                                         pbest_flops] = update_best_particle(
            best_population,
            best_agent_set,
            best_num_parameters,
            best_flops,
            gbest=[
                gbest_individual,
                gbest_agent,
                gbest_params,
                gbest_flops],
            pbest=[
                pbest_individuals,
                pbest_agent_Set,
                pbest_params,
                pbest_flops])
        Log.info('第%d代的gbest_agent' % (curr_gen) + str(gbest_agent))
        Log.info('第%d代的pbest_agent_Set' % (curr_gen) + str(np.mean(pbest_agent_Set)))
        Log.info('EVOLVE[%d-gen]-Finish the updating（当前代个体最佳解、全局最佳解更新完成）' % (curr_gen))

        # 第二代及其后面所有代的种群根据当前的种群 错误率 参数 浮点运算数 代数进行结果保存
        Utils.save_population_and_agent('populations', best_population, best_agent_set, best_num_parameters,
                                            best_flops,
                                            curr_gen)
        Utils.save_population_and_agent('pbest', pbest_individuals, pbest_agent_Set, pbest_params, pbest_flops,
                                            curr_gen)
        Utils.save_population_and_agent('gbest', [gbest_individual], [gbest_agent], [gbest_params],
                                            [gbest_flops], curr_gen)

    end = time.time()
    Log.info('Total Search Time: %.2f seconds' % (end - start))
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    Log.info("%02dh:%02dm:%02ds" % (h, m, s))

    # 权重查看
    # Log.info('运行后pso惯性权重：' + str(weight_set))
    # Log.info('运行后c1权重：' + str(c1_set))
    # Log.info('运行后c2权重：' + str(c2_set))

    # 最后保存的搜索时间 gpu的序号
    search_time = str("%02dh:%02dm:%02ds" % (h, m, s))
    equipped_gpu_ids, _ = GPUTools._get_equipped_gpu_ids_and_used_gpu_info()
    num_GPUs = len(equipped_gpu_ids)

    # 最后的全局错误率就是所求的最后的全局错误率
    proxy_err = copy.deepcopy(gbest_agent)

    # final training and test on testset
    gbest_agent, num_parameters, flops = fitness_test(gbest_individual)
    Log.info('Error=[%.5f], #parameters=[%d], FLOPs=[%d]' % (gbest_agent, gbest_params, gbest_flops))
    Utils.save_population_and_agent('final_gbest', [gbest_individual], [gbest_agent], [num_parameters], [flops],
                                        -1,
                                        proxy_err, search_time + ', GPUs:%d' % num_GPUs)


def __read_ini_file(section, key):
    config = configparser.ConfigParser()
    config.read('global.ini')
    return config.get(section, key)


if __name__ == '__main__':
    import warnings

    # 忽略所有警告
    warnings.filterwarnings("ignore")

    create_directory()
    params = Utils.get_init_params()
    start_init_gen = int(__read_ini_file('PSO', 'start_init_gen'))
    weight = float(__read_ini_file('PSO', 'weight'))
    c1 = float(__read_ini_file('PSO', 'c1'))
    c2 = float(__read_ini_file('PSO', 'c2'))
    batch_size = int(__read_ini_file('SEARCH', 'batch_size'))
    weight_decay = float(__read_ini_file('SEARCH', 'weight_decay'))
    Tp = float(__read_ini_file('SEARCH', 'Tp'))
    Tf = float(__read_ini_file('SEARCH', 'Tf'))
    wp = list(map(float, __read_ini_file('SEARCH', 'wp').split(',')))
    wf = list(map(float, __read_ini_file('SEARCH', 'wf').split(',')))

    evolveCNN(params)
