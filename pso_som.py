import numpy as np
import pandas as pd
import os
import gc
import myfunc
import re
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import numpy as np
import pylab as pl

#
# class SOM(object):
#     def __init__(self, X, output, iteration, batch_size):
#         """
#         :param X:  形状是N*D， 输入样本有N个,每个D维
#         :param output: (n,m)一个元组，为输出层的形状是一个n*m的二维矩阵
#         :param iteration:迭代次数
#         :param batch_size:每次迭代时的样本数量
#         初始化一个权值矩阵，形状为D*(n*m)，即有n*m权值向量，每个D维
#         """
#         self.X = X
#         self.output = output
#         self.iteration = iteration
#         self.batch_size = batch_size
#         self.W = np.random.rand(X.shape[1], output[0] * output[1])
#         print(self.W.shape)
#
#     def GetN(self, t):
#         """
#         :param t:时间t, 这里用迭代次数来表示时间
#         :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
#         """
#         a = min(self.output)
#         return int(a - float(a) * t / self.iteration)
#
#     def Geteta(self, t, n):
#         """
#         :param t: 时间t, 这里用迭代次数来表示时间
#         :param n: 拓扑距离
#         :return: 返回学习率，
#         """
#         return np.power(np.e, -n) / (t + 2)
#
#     def updata_W(self, X, t, winner):
#         N = self.GetN(t)
#         for x, i in enumerate(winner):
#             to_update = self.getneighbor(i[0], N)
#             for j in range(N + 1):
#                 e = self.Geteta(t, j)
#                 for w in to_update[j]:
#                     self.W[:, w] = np.add(self.W[:, w], e * (X[x, :] - self.W[:, w]))
#
#     def getneighbor(self, index, N):
#         """
#         :param index:获胜神经元的下标
#         :param N: 邻域半径
#         :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
#         """
#         a, b = self.output
#         length = a * b
#
#         def distence(index1, index2):
#             i1_a, i1_b = index1 // a, index1 % b
#             i2_a, i2_b = index2 // a, index2 % b
#             return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)
#
#         ans = [set() for i in range(N + 1)]
#         for i in range(length):
#             dist_a, dist_b = distence(i, index)
#             if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
#         return ans
#
#     def train(self):
#         """
#         train_Y:训练样本与形状为batch_size*(n*m)
#         winner:一个一维向量，batch_size个获胜神经元的下标
#         :return:返回值是调整后的W
#         """
#         count = 0
#         while self.iteration > count:
#             train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
#             normal_W(self.W)
#             normal_X(train_X)
#             train_Y = train_X.dot(self.W)
#             winner = np.argmax(train_Y, axis=1).tolist()
#             self.updata_W(train_X, count, winner)
#             count += 1
#         return self.W
#
#     def train_result(self):
#         normal_X(self.X)
#         train_Y = self.X.dot(self.W)
#         winner = np.argmax(train_Y, axis=1).tolist()
#         print(winner)
#         return winner
#
#
# def normal_X(X):
#     """
#     :param X:二维矩阵，N*D，N个D维的数据
#     :return: 将X归一化的结果
#     """
#     N, D = X.shape
#     for i in range(N):
#         temp = np.sum(np.multiply(X[i], X[i]))
#         X[i] /= np.sqrt(temp)
#     return X
#
#
# def normal_W(W):
#     """
#     :param W:二维矩阵，D*(n*m)，D个n*m维的数据
#     :return: 将W归一化的结果
#     """
#     for i in range(W.shape[1]):
#         temp = np.sum(np.multiply(W[:, i], W[:, i]))
#         W[:, i] /= np.sqrt(temp)
#     return W
#
#
# # 画图
# def draw(C):
#     colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
#     for i in range(len(C)):
#         coo_X = []  # x坐标列表
#         coo_Y = []  # y坐标列表
#         for j in range(len(C[i])):
#             coo_X.append(C[i][j][0])
#             coo_Y.append(C[i][j][1])
#         pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)
#
#     pl.legend(loc='upper right')
#     pl.show()
#
#
# #################### dataset 是 N*d 的mat文件：样本量*指标
# dataset = np.mat()
# dataset_old = dataset.copy()
#
# som = SOM(dataset, (5, 5), 1, 30)
# som.train()
# res = som.train_result()
# classify = {}
# for i, win in enumerate(res):
#     if not classify.get(win[0]):
#         classify.setdefault(win[0], [i])
#     else:
#         classify[win[0]].append(i)
# C = []  # 未归一化的数据分类结果
# D = []  # 归一化的数据分类结果
# for i in classify.values():
#     C.append(dataset_old[i].tolist())
#     D.append(dataset[i].tolist())
# draw(C)
# draw(D)
#
# ###########################################
# """
# 粒子群算法求解函数最大值（最小值）
# f(x)= x + 10*sin5x + 7*cos4x
# """
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 粒子
# class Particle:
#     def __init__(self):
#         self.p = 0  # 粒子当前位置
#         self.v = 0  # 粒子当前速度
#         self.pbest = 0  # 粒子历史最好位置
#
#
# class PSO:
#     def __init__(self, N=20, iter_N=100):
#         self.w = 0.2  # 惯性因子
#         self.c1 = 1  # 自我认知学习因子
#         self.c2 = 2  # 社会认知学习因子
#         self.gbest = 0  # 种群当前最好位置
#         self.N = N  # 种群中粒子数量
#         self.POP = []  # 种群
#         self.iter_N = iter_N  # 迭代次数
#
#     # 适应度值计算函数
#     def fitness(self, p):
#         dis = EuclideanDistances(X, bird.p)
#         # winner = np.argmax(dis, axis=1).tolist()
#         return dis.min(axis=1).sum()
#
#     # 找到全局最优解
#     def g_best(self, pop):
#         for bird in pop:
#             if bird.fitness > self.fitness(self.gbest):
#                 self.gbest = bird.p
#
#     # 初始化种群
#     def initPopulation(self, pop, N):
#         for i in range(N):
#             bird = Particle()
#             bird.p = np.random.uniform(-1, 1, (D, m)).T
#             bird.fitness = self.fitness(bird.p)
#             bird.pbest = bird.fitness
#             pop.append(bird)
#
#         # 找到种群中的最优位置
#         self.g_best(pop)
#
#     # 更新速度和位置
#     def update(self, pop):
#         for bird in pop:
#             v = self.w * bird.v + self.c1 * np.random.random() * (
#                     bird.pbest - bird.p) + self.c2 * np.random.random() * (self.gbest - bird.p)
#
#             p = bird.p + v
#
#             if -10 < p < 10:
#                 bird.p = p
#                 bird.v = v
#                 # 更新适应度
#                 bird.fitness = self.fitness(bird.p)
#
#                 # 是否需要更新本粒子历史最好位置
#                 if bird.fitness > self.fitness(bird.pbest):
#                     bird.pbest = bird.p
#
#
# t = pso.POP[1]
# pso.c1
# v = pso.w * t.v + pso.c1 * np.random.random() * (
#         t.pbest - t.p) + pso.c2 * np.random.random() * (pso.gbest - t.p)
#
# p = t.p + v
#
#
# def implement(self):
#     # 初始化种群
#     self.initPopulation(self.POP, self.N)
#
#     # 迭代
#     for i in range(self.iter_N):
#         # 更新速度和位置
#         self.update(self.POP)
#
#         # 更新种群中最好位置
#         self.g_best(self.POP)
#
#
# pso.initPopulation(pso.POP, pso.N)
#
# plt.close('all')
# pso = PSO(N=20, iter_N=50)
# pso.implement()
#
# for ind in pso.POP:
#     print("x = ", ind.p, "f(x) = ", ind.fitness)
#
# print("最优解 x = ", pso.gbest, "相应最大值 f(x) = ", pso.fitness(pso.gbest))
#
# plt.show()

############################################################
###############先 2 个指标，寻找最优的 5 个中心点
data4= data4.dropna()
# 维度D  样本量N 神经元个数m 步数 max_steps
m = 5;max_steps=100;
var = ['iyear','suicide','propextent','multiple','INT_ANY','总体评分','nkill', 'nwound']
a=data4[var]
a['propextent']=[5-x if x >0 else x for x in a['propextent']]
D = len(var)



d = a.sort_values('nkill')    .tail(1000)
d = np.mat(d.values)
N = len(d)

# 归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
d2 = min_max_scaler.fit_transform(d)
# 正太化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(d)
X = scaler.transform(d)

X.shape
myfunc.search('font')
################绘制频率直方图
t=X[:,-1];t=t[t<0.3]
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))
ax0.hist(t,80,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
##pdf概率分布图，一万个数落在某个区间内的数有多少个
ax0.set_title('恐怖袭击死亡人数pdf',fontproperties=font_set)
ax1.hist(t,20,normed=1,histtype='bar',facecolor='pink',alpha=0.75,cumulative=True,rwidth=0.8)
#cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
ax1.set_title("恐怖袭击死亡人数cdf",fontproperties=font_set)
fig.subplots_adjust(hspace=0.4)
plt.show()

# 单个粒子携带信息为5个中心点的坐标集合： D*m 如：2*5

def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A, BT)
    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


class PSO(object):
    def __init__(self, population_size, max_steps):
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 2
        self.population_size = population_size  # 粒子群数量
        self.dim = D*m  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [-10, 10]  # 解空间范围
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                   (self.population_size, self.dim))  # 初始化粒子群位置
        self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速度
        fitness = self.calculate_fitness(self.x)
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置
        self.pg_path = self.pg.reshape(1,D*m)  # 全局最佳位置演变
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.max(fitness)  # 全局最佳适应度
        self.global_best_fitness_path = [self.global_best_fitness] # 全局最佳适应度

    # 适应度值计算函数
    def calculate_fitness(self, x):
        fitness=[]
        for iter in x:
            t = iter.reshape(m, D)
            dis = EuclideanDistances(X, t).min(axis=1).sum()
            fitness = fitness  +[dis]
            # winner = np.argmax(dis, axis=1).tolist()
        return np.array(fitness)

    def evolve(self):
        # fig = plt.figure()
        for step in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            self.x = self.v + self.x
            fitness = self.calculate_fitness(self.x)
            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            self.pg_path = np.vstack((self.pg_path,self.pg))
            self.global_best_fitness_path = self.global_best_fitness_path  + [self.global_best_fitness]
            # if step%200==0 or step<500:
            #     # print(self.global_best_fitness)
            #     print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.mean(fitness)))


pso = PSO(D*m, 50)
pso.evolve()

neuron = pso.pg.reshape(m,D)

pso.global_best_fitness
pso.pg.reshape(m,D)

##导出路径
a1=pso.pg_path
a2=pso.global_best_fitness_path
a3=pd.DataFrame(a2)
a4=pd.DataFrame(a1)
a4 = np.log(a4)
a3.to_csv('e:\\jm\\temp.csv')
a4.to_csv('e:\\jm\\temp1.csv')
########################################
########################################

class SOM(object):
    def __init__(self, X, output, iteration, batch_size):
        """
        :param X:  形状是N*D， 输入样本有N个,每个D维
        :param output: (n,m)一个元组，为输出层的形状是一个n*m的二维矩阵
        :param iteration:迭代次数
        :param batch_size:每次迭代时的样本数量
        初始化一个权值矩阵，形状为D*(n*m)，即有n*m权值向量，每个D维
        """
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        # self.W = np.random.rand(X.shape[1], output[0] * output[1])
        t = np.random.rand(output[0] * output[1]-5,X.shape[1])
        t = np.vstack((t,neuron))
        np.random.shuffle(t)
        self.W = t.T
        print(self.W.shape)

    def GetN(self, t):
        """
        :param t:时间t, 这里用迭代次数来表示时间
        :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
        """
        a = min(self.output)
        return int(a - float(a) * t / self.iteration)

    def Geteta(self, t, n):
        """
        :param t: 时间t, 这里用迭代次数来表示时间
        :param n: 拓扑距离
        :return: 返回学习率，
        """
        return np.power(np.e, -n) / (t + 2)

    def updata_W(self, X, t, winner):
        N = self.GetN(t)
        for x, i in enumerate(winner):
            to_update = self.getneighbor(i[0], N)
            for j in range(N + 1):
                e = self.Geteta(t, j)
                for w in to_update[j]:
                    self.W[:, w] = np.add(self.W[:, w], e * (X[x, :] - self.W[:, w]))

    def getneighbor(self, index, N):
        """
        :param index:获胜神经元的下标
        :param N: 邻域半径
        :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
        """
        a, b = self.output
        length = a * b

        def distence(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b
            i2_a, i2_b = index2 // a, index2 % b
            return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)

        ans = [set() for i in range(N + 1)]
        for i in range(length):
            dist_a, dist_b = distence(i, index)
            if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
        return ans

    def train(self):
        """
        train_Y:训练样本与形状为batch_size*(n*m)
        winner:一个一维向量，batch_size个获胜神经元的下标
        :return:返回值是调整后的W
        """
        count = 0
        while self.iteration > count:
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
            normal_W(self.W)
            normal_X(train_X)
            train_Y = train_X.dot(self.W)
            winner = np.argmax(train_Y, axis=1).tolist()
            self.updata_W(train_X, count, winner)
            count += 1
        return self.W

    def train_result(self):
        normal_X(self.X)
        train_Y = self.X.dot(self.W)
        winner = np.argmax(train_Y, axis=1).tolist()
        print(winner)
        return winner

def normal_X(X):
    """
    :param X:二维矩阵，N*D，N个D维的数据
    :return: 将X归一化的结果
    """
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X

def normal_W(W):
    """
    :param W:二维矩阵，D*(n*m)，D个n*m维的数据
    :return: 将W归一化的结果
    """
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:, i], W[:, i]))
        W[:, i] /= np.sqrt(temp)
    return W

# 画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)

    pl.legend(loc='upper right')
    pl.show()
################################
dataset = X
dataset = np.mat(dataset)
dataset_old = dataset.copy()

som = SOM(dataset, (10,10), 50, 2000)
som.train()
res = som.train_result()
len(res)

a=[x[0] for x in res]
b=data4.eventid.to_frame()
b['n']=a
b.to_csv('e:\\jm\\temp4.csv')
som.W.shape
X.shape

classify = {}
for i, win in enumerate(res):
    if not classify.get(win[0]):
        classify.setdefault(win[0], [i])
    else:
        classify[win[0]].append(i)
len(classify)


t1 = []
for t in classify.keys():
    t1 = t1 + [len(classify[t])]
t2=classify.keys()

t=pd.DataFrame()
t['t1']=t1;t['t2']=t2;
t=t.sort_values('t1')


n=10
x=[x%n for x  in t2]
y=[int(x/n) for x  in t2]
df=pd.DataFrame();df['x']=x;df['y']=y;df['z']=t1
df=df.pivot(index='x',columns='y').fillna(0).stack().reset_index()
x=df.x;y=df.y;z=df.z
df.sort_values('z')

C = []  # 未归一化的数据分类结果
D = []  # 归一化的数据分类结果
for i in classify.values():
    C.append(dataset_old[i].tolist())
    D.append(dataset[i].tolist())
draw(C)
draw(D)
np.exp(0.02626)

t_1=10
x=[1,2,3,4,5]*t_1;y=np.array([1,2,3,4,5]).repeat(t_1).tolist()

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

z=np.array(t1)
N = int(len(z)**.5)
n=10
z = z.reshape(n, n)
plt.imshow(z+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
        cmap=cm.hot, norm=LogNorm())
plt.colorbar()
plt.show()
################################################
###################支持向量机###################
from sklearn import svm
from sklearn.metrics import accuracy_score,show_accuracy
import random
random.randint(1, 1000)
t = np.random.randint( 1,1000, size = 500 )
# data_x = X[-50:]

data_x = X[t]
data_y=np.array([x[0] for x in np.array(res)[t]])

t1 = [x for x in range(len(data_x)) if (data_x[x,0]<0.3) & (data_x[x,1]<0.3)]
data_x = data_x[t1]
data_y = data_y[t1]


# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(data_x, data_y)

clf.score(data_x, data_y)  # 精度
y_hat = clf.predict(data_x)
accuracy_score(y_hat, data_y, '训练集')

x1_min, x1_max = data_x[:, 0].min(), data_x[:, 0].max()  # 第0列的范围
x2_min, x2_max = data_x[:, 1].min(), data_x[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# print 'grid_test = \n', grid_testgrid_hat = clf.predict(grid_test)       # 预测分类值grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

# Z = clf.decision_function(grid_test)
# Z = Z[:, 0].reshape(x1.shape)
grid_hat = clf.predict(grid_test)
grid_hat = grid_hat.reshape(x1.shape)

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
plt.scatter(data_x[:, 0], data_x[:, 1], c='y', edgecolors='k', s=50,facecolors='none', cmap=cm_dark)  # 样本
# plt.scatter(data_x[:, 0], data_x[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'nkill', fontsize=13)
plt.ylabel(u'nwound', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'分布状况', fontsize=15)
# plt.grid()
plt.show()

'decision_function:\n', clf.decision_function(data_x)
'\npredict:\n', clf.predict(data_x)

data3['nkill'].describe()
np.percentile(data3['nkill'], 99)

from sklearn.metrics import classification_report
# 输出更加详细的其他评价分类性能的指标。
print classification_report(y_test, y_count_predict, target_names = news.target_names)

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

#准备训练样本
x=[[1,8],[3,20],[1,15],[3,35],[5,35],[4,40],[7,80],[6,49]]
y=[1,1,-1,-1,1,-1,-1,1]





##开始训练
clf=svm.SVC()  ##默认参数：kernel='rbf'
clf.fit(x,y)

#print("预测...")
#res=clf.predict([[2,2]])  ##两个方括号表面传入的参数是矩阵而不是list

##根据训练出的模型绘制样本点
for i in x:
    res=clf.predict(np.array(i).reshape(1, -1))
    if res > 0:
        plt.scatter(i[0],i[1],c='r',marker='*')
    else :
        plt.scatter(i[0],i[1],c='g',marker='*')

##生成随机实验数据(15行2列)
rdm_arr=np.random.randint(1, 15, size=(15,2))
##回执实验数据点
for i in rdm_arr:
    res=clf.predict(np.array(i).reshape(1, -1))
    if res > 0:
        plt.scatter(i[0],i[1],c='r',marker='.')
    else :
        plt.scatter(i[0],i[1],c='g',marker='.')
##显示绘图结果
plt.show()