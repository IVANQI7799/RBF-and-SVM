from scipy.io import loadmat
from sklearn import model_selection
from sklearn.cluster import KMeans
from scipy.linalg import norm
import numpy as np
import scipy.io as scio
import math


data= loadmat("data_train.mat")["data_train"]  # 330*33
label = loadmat("label_train.mat")["label_train"]  # 330*1
data_train, data_test, label_train, label_test = model_selection.train_test_split(data, label, random_state=1, test_size=0.2)
#data_test = loadmat("data_test.mat")["data_test"]

center_num = 10
model = KMeans(n_clusters=center_num, max_iter=1500)
model.fit(data_train)
centers = model.cluster_centers_
a=[]

class RBF(object):
    global center_num

    def __init__(self, centers, input_data, label, test_data):
        self.input_data = input_data
        self.label = label
        self.test_data = test_data
        self.centers = centers
        self.w = np.array([np.random.rand(center_num + 1)])
        self.sigma = self.getSigma(self.centers)
        self.threshold = 0

    def getSigma(self, centers):
        sigma_ = 0
        for i in centers:
            for j in centers:
                if abs(norm(i) - norm(j)) > sigma_:
                    sigma_ = abs(norm(i) - norm(j))/math.sqrt(20)
        return sigma_

    def basis(self, data, center):
        # sigma = 0.707
        return math.exp(-norm(data - center) ** 2 / (2 * self.sigma ** 2))

    def calcAct(self, input_data):
        act_val = np.zeros((input_data.shape[0], center_num))
        for xi, x in enumerate(input_data):
            for ci, c in enumerate(self.centers):
                act_val[xi][ci] = self.basis(x, c)
        return act_val


    def acc(self, output, threshold, label, training=True):
        res = np.ones(output.shape)
        res[output > threshold] = 1
        res[output < threshold] = -1
        if training:
            acc= sum(res == label) / output.shape[0]
            print("训练集精准度：")
            print(acc)
            return sum(res == label) / output.shape[0]
        else:
            return res

    def train(self):
        act_val = self.calcAct(self.input_data)
        w0_column = np.array([1 for _ in range(len(act_val))])
        act_val = np.insert(act_val, len(act_val[0]), values=w0_column, axis=1)
        self.w = np.linalg.inv(act_val.T.dot(act_val)).dot(act_val.T).dot(self.label)
        output = act_val.dot(self.w)
        #self.getThreshold(output, label=self.label)
        return output

    def valid(self, data, label):
        act_val_test = self.calcAct(data)
        w0_column = np.array([1 for _ in range(len(act_val_test))])
        act_val_test = np.insert(act_val_test, len(act_val_test[0]), values=w0_column, axis=1)
        output = act_val_test.dot(self.w)
        #self.getThreshold(output,label)
        return output, self.acc(output, self.threshold, label) #output表示输出的label矩阵

    def test(self):
        act_val_test = self.calcAct(self.test_data)
        w0_column = np.array([1 for _ in range(len(act_val_test))])
        act_val_test = np.insert(act_val_test, len(act_val_test[0]), values=w0_column, axis=1)
        output = act_val_test.dot(self.w)
        #print('Label of testing data are:')
        #print(np.squeeze(self.acc(output, self.threshold, None, False)))
        for i in range(66):
            if(output[i]>=0): a.append(1)
            if(output[i] < 0): a.append(-1)
        #print(a)
        return output, self.acc(output, self.threshold, None, False)



network = RBF(centers, data_train, label_train, data_test)
output = network.train()
#ceshi=network.valid(data_train,label_train)
test=network.test()
ture=0
false=0
for i in range(66):
    if(a[i]==label_test[i]): ture+=1
    else: false+=1
acc=ture/(ture+false)
print("验证集精准度：")
print(acc)


