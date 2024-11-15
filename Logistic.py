import random
from scipy.stats import beta
import math
import matplotlib.pyplot as plt
import numpy as np


# 生成并打印随机数
def random_excluding_specific_values():
    # 生成一个 [0, 1) 范围内的随机数
    random_number = random.uniform(0, 1)

    # 检查并排除特定的值
    while random_number in (0, 0.25, 0.75, 1):
        random_number = random.uniform(0, 1)

    return random_number


# 混沌惯性权重
def logistic_map(w_max, w_min, T_max, iterations, sigma, b1, b2):
    # 初始化 r 的初始值
    r = random_excluding_specific_values()
    # print('混沌映射的初始值为：' + str(r))
    # 初始化惯性权重序列
    w_sequence = []

    for t in range(0, T_max):
        # 应用 Logistic 映射更新 r
        r = 4 * r * (1 - r)
        # 确保 r 在 [0, 1] 范围内
        r = max(0, min(1, r))
        # print('t:' + str(t) + '混沌映射值为：' + str(r))
        # 计算惯性权重 w
        w = r * w_min + (w_max - w_min) * t / T_max
        w = (1 - w) + sigma * beta.rvs(b1, b2)
        # 将计算出的 w 添加到序列中

        w_sequence.append(w)
        # if t == iterations:
        #     return w
    return w_sequence


def plot(w_sequence):
    # 创建一个图表
    plt.figure()
    # 绘制 w_sequence 序列
    plt.plot(w_sequence, label='Inertia Weight')
    # 添加图例
    plt.legend(loc='lower right')
    # 添加标题和轴标签
    plt.title('Logistic Inertia Weight with Beta Distribution Over Iterations', fontsize=13)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Inertia Weight (w)', fontsize=16, )
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    # 显示图表
    plt.savefig('Logistic Inertia Weight with Beta Distribution Over Iterations.jpg', bbox_inches='tight', dpi=1024)
    plt.show()


# 参数设置
w_max = 0.9  # 惯性权重的最大值
w_min = 0.4  # 惯性权重的最小值
T_max = 40  # 最大迭代次数
iterations = 39  # 迭代次数
sigma = 0.03
b1 = 1
b2 = 2

# 计算惯性权重序列
w = logistic_map(w_max, w_min, T_max, iterations, sigma, b1, b2)
# print(w)
plot(w)


def dynamic_learning_factors(k, max_iter, c_max, c_min):
    # 计算学习因子 c1 和 c2
    c1 = c_max - (c_max - c_min) * math.sin(math.pi * k / (2 * max_iter))
    c2 = c_min + (c_max - c_min) * math.sin(math.pi * k / (2 * max_iter))
    return c1, c2


# 参数设置
k = 50  # 当前迭代次数
max_iter = 100  # 最大迭代次数
c_max = 2.01  # 学习因子最大值
c_min = 0.8  # 学习因子最小值


# 计算并打印动态学习因子
# c1, c2 = dynamic_learning_factors(k, max_iter, c_max, c_min)
# print(f"Dynamic Learning Factors at iteration {k}: c1 = {c1}, c2 = {c2}")

# 假设的适应度函数
def objective_function(position):
    # 这里应该是实际的优化问题的目标函数
    # 为了示例，我们使用一个简单的二次函数
    return sum(x ** 2 for x in position)


# 计算 pi
def calculate_pi(particle, all_particles, N):
    fitness_particle = objective_function(particle)
    fitness_sum = sum(objective_function(p) for p in all_particles)

    # 计算 pi 的分子
    numerator = math.exp(fitness_particle)

    # 计算 pi 的分母
    denominator = math.exp(fitness_sum / N)

    # 计算 pi
    pi = numerator / denominator
    return pi


def pso_velocity_update(pi, m, beta, delta, w, c1, c2, r1, r2, pbest, gbest, x, v):
    if pi < 0.6:
        # Update velocity 径直走
        v_new = w * v + math.pow(-1, delta) * c1 * r1 * ((pbest + gbest) / 2 - m * x) + math.pow(-1,
                                                                                                 delta) * c2 * r2 * (
                        (pbest - gbest) / 2 - m * x)
        # Update position 全局
        x_new = w * x + (1 - w) * v_new

    elif pi >= 0.6:
        # Update velocity 绕圈走
        v_new = w * v + math.pow(-1, delta) * c1 * r1 * (
                (pbest + gbest) / 2 - math.cos(2 * np.pi * beta) * x) + math.pow(-1, delta) * c2 * r2 * (
                        (pbest - gbest) / 2 - math.cos(2 * np.pi * beta) * x)
        # Update position 局部
        x_new = x + v_new

    return v_new, x_new


# Example usage:
# Initialize parameters
w = 0.9  # Inertia weight
c1 = 2.0  # Cognitive learning factor
c2 = 2.0  # Social learning factor
# Random numbers
r1 = np.random.rand()
r2 = np.random.rand()
# delta rand choice
delta = selected = random.choice([-1, 0])
print(delta)
m = random.uniform(0.4, 0.9)
print(m)
beta = random.random()
print(beta)
pbest = np.array([0.0, 0.0])  # Individual best position
gbest = np.array([1.0, 1.0])  # Global best position

x = np.array([0.001, 0.0009])  # Current position
# 或者如果你想每个数组有不同的值，可以这样操作：
values = [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
arrays = [np.array(value) for value in values]

v = np.array([0.0, 0.0])  # Current velocity

# pi = calculate_pi(x, arrays, 3)
# print(pi)
# # Update velocity and position
# v_new, x_new = pso_velocity_update(pi, m, beta, delta, w, c1, c2, r1, r2, pbest, gbest, x, v)
#
# print("Updated velocity:", v_new)
# print("Updated position:", x_new)
