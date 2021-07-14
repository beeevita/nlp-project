#coding=utf8
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from numpy import *



class logistic(object):
    def __init__(self, lr, epoch,batch_size,shuffle):
        self.W = None  # 初始化
        self.lr = lr
        self.epoch = epoch
        self.batch_size=batch_size
        self.shuffle=shuffle

    def sigmoid(self, x):  # sigmoid函数
        return 1.0 / (1 + np.exp(-x))

    def get_mini_batches(self,X, Y, mini_batch_size,seed=0):
        np.random.seed(seed)
        total = X.shape[0]
        Y = np.array(Y).reshape(Y.shape[0], 1)

        mini_batches = []
        # print(Y.shape)
        # step1：打乱训练集
        # 生成0~m-1随机顺序的值，作为下标
        if self.shuffle:
            shuffle_index = list(np.random.permutation(total))
        else:
            shuffle_index = [i for i in range(0, total)]
        # print(shuffle_index)
        # 打乱后的训练集
        shuffled_X = X[shuffle_index, :]  # 取所有列，打乱后的顺序
        # print(shuffled_X.shape)    # 1284848 16532  一个稀疏矩阵
        shuffled_Y = Y[shuffle_index, :]
        # step2：按照batchsize分割训练集
        # 得到总的子集数目，math.floor表示向下取整
        batches_num = math.floor(total / mini_batch_size)
        for k in range(0, batches_num):
            # 冒号：表示取所有行，第二个参数a：b表示取第a列到b-1列，不包括b
            mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
            mini_batch_Y = mini_batch_Y.reshape(mini_batch_Y.shape[0], )

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # m%mini_batch_size != 0表示还有剩余的不够一个batch大小，把剩下的作为一个batch
        if total % mini_batch_size != 0:
            mini_batch_X = shuffled_X[mini_batch_size * batches_num, :]
            mini_batch_Y = shuffled_Y[mini_batch_size * batches_num, :]
            mini_batch_Y = mini_batch_Y.reshape(mini_batch_Y.shape[0], )

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    # 损失函数
    def loss_func(self, X, y):
        m, n = X.shape
        h = self.sigmoid(X.dot(self.W)) # 经过sigmoid函数处理的预测值，这里用点乘
        # 此处的loss是矩阵类型,为了便于画图将其中的数取出，(似然函数的对数形式)，
        # 需要似然函数取最大值，那么损失函数取似然函数的相反数，所以分母上为-m
        loss = (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))) / (-m)
        dW = X.T.dot((h - y)) / m  # 求导
        return dW,loss

    def BGD(self, X_train,Y_train, X_dev, Y_dev):
        self.W = np.random.uniform(size=X_train.shape[1])
        loss_list = []
        acc_list = []
        for i in range(self.epoch):
            dW,loss = self.loss_func(X_train, Y_train)
            self.W -= self.lr * dW  # 参数更新
            predict_dev_Y = self.predict(X_dev)
            acc = (predict_dev_Y == Y_dev).sum() / Y_dev.shape[0]
            acc_list.append(acc)
            loss_list.append(loss)
            print("Epoch %s, Dev Acc %.3f" % (i,acc))

        return loss_list,acc_list

    def predict(self, X_test):
        y_pred_list = []
        for xx in X_test:
            y_pred = self.sigmoid(xx.dot(self.W))
            # y_pred_list.append(y_pred[0,0])
            if y_pred >= 0.5:
                y_pred_list.append(1)
            else:
                y_pred_list.append(0)
        return y_pred_list


    def SGD(self,X_train,Y_train,X_dev,Y_dev):
        self.W = np.random.uniform(size=X_train.shape[1])
        loss_list = []
        acc_list = []

        for i in range(self.epoch):
            loss=0
            for batch_X, batch_Y in self.get_mini_batches(X_train,Y_train, self.batch_size):
                dW, loss = self.loss_func(batch_X, batch_Y)
                # print(batch_X.shape)   # (1024, 16532)
                # print(batch_Y.shape)  # (1024,)
                # print(self.W.shape)  # (16532,)
                self.W -= self.lr * dW  # 参数更新
            predict_dev_Y = self.predict(X_dev)
            acc = (predict_dev_Y == Y_dev).sum() / Y_dev.shape[0]
            acc_list.append(acc)
            loss_list.append(loss)
            print("Epoch %s, Dev Acc %.3f" % (i, acc))

        return loss_list, acc_list


class softmax_regression():
    def __init__(self, lr, epoch,batch_size,shuffle,n_class):
        self.W = None  # 初始化
        self.lr = lr
        self.epoch = epoch
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.n_class = n_class

    def softmax(self,x):
        return np.exp(x) / np.exp(x).sum(-1, keepdims=True)

    def onehot(self,x, n_class):
        return np.eye(n_class)[x]

    def predict(self, X_test):
        y_pred = self.softmax(X_test.dot(self.W))
        return y_pred.argmax(-1)

    def loss_func(self, X, y):
        m, n = X.shape
        h = self.softmax(X.dot(self.W))
        # 此处的loss是矩阵类型,为了便于画图将其中的数取出，(似然函数的对数形式)，
        # 需要似然函数取最大值，那么损失函数取似然函数的相反数，所以分母上为-m
        dW = -X.T.dot(self.onehot(y, self.n_class)- h)  # 求导

        return dW

    def BGD(self, X_train,Y_train, X_dev, Y_dev):
        self.W = np.random.uniform(size=(X_train.shape[1], self.n_class))
        acc_list = []
        for i in range(self.epoch):
            dW = self.loss_func(X_train, Y_train)
            self.W -= self.lr * dW  # 参数更新
            predict_dev_Y = self.predict(X_dev)
            acc = (predict_dev_Y == Y_dev).sum() / Y_dev.shape[0]
            acc_list.append(acc)
            print("Epoch %s, Dev Acc %.3f" % (i,acc))

        return acc_list

    def get_mini_batches(self,X, Y, mini_batch_size,seed=0):
        np.random.seed(seed)
        total = X.shape[0]
        Y = np.array(Y).reshape(Y.shape[0], 1)
        print(type(X))

        mini_batches = []
        # print(Y.shape)
        # step1：打乱训练集
        # 生成0~m-1随机顺序的值，作为下标
        if self.shuffle:
            shuffle_index = list(np.random.permutation(total))
        else:
            shuffle_index = [i for i in range(0, total)]
        # print(shuffle_index)
        # 打乱后的训练集
        shuffled_X = X[shuffle_index, :]  # 取所有列，打乱后的顺序
        # print(shuffled_X.shape)    # 1284848 16532  一个稀疏矩阵
        shuffled_Y = Y[shuffle_index, :]
        # step2：按照batchsize分割训练集
        # 得到总的子集数目，math.floor表示向下取整
        batches_num = math.floor(total / mini_batch_size)
        for k in range(0, batches_num):
            # 冒号：表示取所有行，第二个参数a：b表示取第a列到b-1列，不包括b
            mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
            mini_batch_Y = mini_batch_Y.reshape(mini_batch_Y.shape[0], )

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # m%mini_batch_size != 0表示还有剩余的不够一个batch大小，把剩下的作为一个batch
        if total % mini_batch_size != 0:
            mini_batch_X = shuffled_X[mini_batch_size * batches_num, :]
            mini_batch_Y = shuffled_Y[mini_batch_size * batches_num, :]
            mini_batch_Y = mini_batch_Y.reshape(mini_batch_Y.shape[0], )

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def SGD(self,X_train,Y_train,X_dev,Y_dev):
        self.W = np.random.uniform(size=(X_train.shape[1], self.n_class))
        acc_list = []

        for i in range(self.epoch):
            for batch_X, batch_Y in self.get_mini_batches(X_train,Y_train, self.batch_size):
                dW = self.loss_func(batch_X, batch_Y)
                self.W -= self.lr * dW  # 参数更新
            predict_dev_Y = self.predict(X_dev)
            acc = (predict_dev_Y == Y_dev).sum() / Y_dev.shape[0]
            acc_list.append(acc)
            print("Epoch %s, Dev Acc %.3f" % (i, acc))

        return acc_list

if __name__=='__main__':
    l = logistic(lr=1, epoch=1,batch_size=1024,shuffle=True)
