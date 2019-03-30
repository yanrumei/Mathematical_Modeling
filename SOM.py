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
        # t = np.random.rand(output[0] * output[1]-5,X.shape[1])
        # t = np.vstack((t,neuron))
        # np.random.shuffle(t)
        self.W = np.random.rand(X.shape[1],output[0] * output[1])
        print(self.W)
        # self.W = t.T
        # print(self.W.shape)
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
        # print(winner)
        for x, i in enumerate(winner):
            to_update = self.getneighbor(i[0], N)
            for j in range(N + 1):
                e = self.Geteta(t, j)
                for w in to_update[j]:
                    # print(self.W[:, w])
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
        # print(winner)
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

############分别是（输入数据，输出层神经，更新次数，每次更新样本量）
dataset = X
dataset = np.mat(dataset)

dataset_old = dataset.copy()

som = SOM(dataset, (5, 1), 10, 500)
som.train()
res = som.train_result()

som.W
# win=res[4]
# len(res)
classify = {}
for i, win in enumerate(res):
    if not classify.get(win[0]):
        classify.setdefault(win[0], [i])
    else:
        classify[win[0]].append(i)

classify.keys()
len(classify[11])
for t in classify.keys():
    print(len(classify[t]))

C = []  # 未归一化的数据分类结果
D = []  # 归一化的数据分类结果
for i in classify.values():
    C.append(dataset_old[i].tolist())
    D.append(dataset[i].tolist())
draw(C)
draw(D)

D.shape
plt.show()


def calculate_fitness(x):
    fitness = []
    for iter in x:
        t = iter.reshape(5, 2)
        dis = EuclideanDistances(X, t).min(axis=1).sum()
        fitness = fitness + [dis]
        # winner = np.argmax(dis, axis=1).tolist()
    return fitness


a=np.greater(np.array([1,2,3,4]),np.array([2,3,4,3]))
b=np.array([1,2,3,4])
b[a]

r1 = np.random.rand(5, 10)
r2 = np.random.rand(5, 10)
fitness = calculate_fitness(x)
update_id = np.greater(individual_best_fitness, fitness)
self.p[update_id] = self.x[update_id]
self.individual_best_fitness[update_id] = fitness[update_id]


bird.p = np.random.uniform(-1, 1, (D, m)).T
bird.fitness = fitness(bird.p)
bird.pbest = bird.fitness
pop.append(bird)

help(np.random.uniform)


def fitness(bird):
    dis = EuclideanDistances(X, bird.p)
    # winner = np.argmax(dis, axis=1).tolist()
    return dis.min(axis=1).sum()


################################################AHP
import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from numpy import *

class AHP:
    def __init__(self, array):
        self.row = len(array)
        self.col = len(array[0])

    def get_tezheng(self, array):# 获取特征值和特征向量
        te_val, te_vector = np.linalg.eig(array)
        list1 = list(te_val)
        print("特征值为：", te_val)
        print("特征向量为：", te_vector)
        # 得到最大特征值对应的特征向量
        max_val = np.max(list1)
        index = list1.index(max_val)
        max_vector = te_vector[:, index]
        print("最大的特征值:" + str(max_val) + "   对应的特征向量为：" + str(max_vector))
        return max_val, max_vector

    def RImatrix(self, n):  # 建立RI矩阵
        print(n)
        n1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        n2 = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45]
        d = dict(zip(n1,n2))
        print("该矩阵在一致性检测时采用的RI值为：", d[n])
        return d[n]

    def test_consitstence(self, max_val, RI):  # 测试一致性
        CI = (max_val - self.row) / (self.row - 1)
        if RI == 0:
            print("判断矩阵的RI值为  " + str(0) + "  通过一致性检验")
            return True
        else:
            CR = CI / RI
            if CR < 0.10:
                print("判断矩阵的CR值为  " + str(CR) + "  通过一致性检验")
                return True

            else:
                print("判断矩阵的CR值为  " + str(CR) + "  判断矩阵未通过一致性检验，请重新输入判断矩阵")
                return False

def main():
    weigh_matrix = []
    num_of_standard = []
    l0 = []
    l1 = []

    def input_data(f1):  # 对输入的判断矩阵进行处理
        l0 = []
        l1 = []
        li = []
        with open(f1) as f:
            reader = csv.reader(f)
            for line in reader:
                print(line)
                if line == []:
                    l1.append(l0)
                    l0 = []
                    continue
                for i in line:
                    if "/" in i:
                        b = i.split("/")
                        li.append(int(b[0]) / float(b[1]))
                    else:
                        li.append(float(i))
                l0.append(li)
                li = []
            l1.append(l0)
        print(l1)
        return l1
    l1= input_data(r"C:\Users\adm\Desktop\data11.txt")
    print("l1",l1)


    def normalize_vector(max_vector):  # 特征向量归一化
        vector_after_normalization = []
        sum0 = np.sum(max_vector)
        for i in range(len(max_vector)):
            vector_after_normalization.append(max_vector[i] / sum0)
        print("该级指标的权重权重矩阵为：  " + str(vector_after_normalization))

        file = open(r"C:\Users\adm\Desktop\data000112.txt", 'a')
        for i in vector_after_normalization:
            file.write(str(i))
            file.write("\n")

        return vector_after_normalization

    def to_input_matrix():#判别矩阵的一致性检验
         for i in l1:
            length = len(l1)
            print(i)
            a = AHP(i)
            max_val, max_vector = a.get_tezheng(i)
            record_max_vector = max_vector
            RI = a.RImatrix(len(i))
            flag = a.test_consitstence(max_val, RI)
            while not flag:
                print("对比矩阵未通过一致性检验，请重新输入对比矩阵！")
                break
                #flag = to_input_matrix(length)
            weight = normalize_vector(record_max_vector)  # 返回权重[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
            weigh_matrix.append(weight)
         print("最终的权重矩阵为：", weigh_matrix)
         return weigh_matrix
    w00=to_input_matrix()
    print("w00",w00)
    def construct_weightmatrix(weight):
        data = []
        l = []
        for i in weight[0]:
            data.append(i.real)
        #print(data)

        for i in weight[1:4]:
            for j in i:
                data.append(j.real)

        for i in weight[4:]:
            for j in i:
                data.append(j.real)
        print(data)
        data2 = data
        print(len(data2))
        return data2
    d = construct_weightmatrix(w00)
    print(d)

    def get_index(data ):  # 第一个矩阵1*3，第二个矩阵7*3。第三个矩阵16*7
        row = [3, 7, 16]
        column = [1, 3, 7]

        x1 = [0, 1, 2]
        y1 = [0, 0, 0]

        x2 = [0, 1, 2, 3, 4, 5, 6]
        y2 = [0, 0, 1, 1, 1, 2, 2]

        x3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ]
        y3 = [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6]
        weight = []

        def transform(x, y, data, row, col):
            h = list(zip(x, y))
            print(h)
            mat1 = zeros((row, col))
            for i in range(len(data)):
                mat1[h[i]] = data[i]
            weight.append(mat1)
            print(weight)

        transform(x1, y1, data[:3], 3, 1)
        transform(x2, y2, data[3:10], 7, 3)
        transform(x3, y3, data[10:], 16, 7)
        return weight
    w1 = get_index(d)#返回组织好的权重矩阵d
    for i in w1:
        print(i)
        # print(w1)

    def get_finalweight(w):#矩阵相乘
        j = 0
        s = np.array([1])
        print(np.dot(w[1], w[0]))

        while j < len(w):
            print("weight0[i]", w[j])
            s = np.dot(w[j], s)
            print("s", s)
            j += 1
        print("s", s)
        return s

    s1=get_finalweight(w1)

    def data_process():
        i = 0
        data = []
        global name
        name = []
        with open(r'C:\Users\adm\Desktop\数据.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                else:
                    data.append(row[:])
                    name.append(row[1])

        data = np.asarray(data)
        data1 = data[:, 3:].astype('float64')
        scaler = MinMaxScaler(feature_range=(0, 1))
        data1 = scaler.fit_transform(data1)
        print(len(data1))
        return data1

    def ScoreCalculator():
        SCORE = []
        score = []
        value = data_process()
        print(len(value))
        for i in range(len(value)):
            #print("i",len(value[i]))
            #print(len(s1))
            SCORE = np.sum([x * y for x, y in zip(value[i], s1)]) * 100
            score.append(SCORE)
        score = [i / np.max(score) * 100 for i in score]
        print(score)

        d = dict(zip(name, score))
        d1 = sorted(d.items(), key=lambda items: items[1], reverse=True)
        # print(d1)
        for key, value in d1[:80]:

            if value > 80:
                print(str(key) + "  最终得到的评分为： " + str(value) + "    合理")
            elif 70 < value < 80:
                print(str(key) + "  最终得到的评分为： " + str(value) + "    较为合理")
            elif 60 < value < 70:
                print(str(key) + "  最终得到的评分为： " + str(value) + "    一般合理")
            else:
                print(str(key) + "  最终得到的评分为： " + str(value) + "    不合理")

        print("所有指标权重的和", np.sum(s1))
    ScoreCalculator()

if __name__ == "__main__":
    main()


class SOM(object):
    def __init__(self, X, output, iteration, batch_size):
        """
        初始化一个权值矩阵，形状为D*(n*m)，即有n*m权值向量，每个D维
        """
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        t = np.random.rand(output[0] * output[1]-5,X.shape[1])
        t = np.vstack((t,neuron))
        np.random.shuffle(t)
        self.W = t.T
        print(self.W.shape)

    def GetN(self, t): #返回拓扑距离
        a = min(self.output)
        return int(a - float(a) * t / self.iteration)

    def Geteta(self, t, n):#返回学习率
        return np.power(np.e, -n) / (t + 2)

    def updata_W(self, X, t, winner):
        N = self.GetN(t)
        for x, i in enumerate(winner):
            to_update = self.getneighbor(i[0], N)
            for j in range(N + 1):
                e = self.Geteta(t, j)
                for w in to_update[j]:
                    self.W[:, w] = np.add(self.W[:, w], e * (X[x, :] - self.W[:, w]))

    def getneighbor(self, index, N): #返回相邻神经元
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

    def train(self): #训练神经元
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
    def train_result(self): #返回训练结果
        normal_X(self.X)
        train_Y = self.X.dot(self.W)
        winner = np.argmax(train_Y, axis=1).tolist()
        print(winner)
        return winner