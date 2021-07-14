#coding=utf8

import numpy as np
from numpy import *
import csv
import random
from ngram import Ngram
import pandas as pd
from models import logistic,softmax_regression
import math
from utils import logistic_plot, softmax_plot

EPOCH = 1
SHUFFLE =True
LR = 0.00005
BATCH_SIZE=1024
MODE = 'mini'  # 'SGD', 'BGD' or 'mini'
classifier = 'softmax'  # 'softmax' or 'logistic'


def split_dataset(X, Y, train_rate,test_rate, shuffle=SHUFFLE):
    assert X.shape[0] == Y.shape[0]
    assert train_rate + test_rate <= 1.0 and train_rate + test_rate > 0

    total = X.shape[0]
    index = [i for i in range(total)]
    if shuffle:
        random.shuffle(index)
    train_cnt = int(train_rate * total)
    test_cnt = int(test_rate * total)
    dev_cnt = total - train_cnt - test_cnt

    X_train = X[index[:train_cnt]]
    X_dev = X[index[train_cnt:train_cnt+dev_cnt]]
    X_test = X[index[:test_cnt]]

    Y_train = Y[index[:train_cnt]]
    Y_dev = Y[index[train_cnt:train_cnt+dev_cnt]]
    Y_test = Y[index[:test_cnt]]

    print("Size of trainset: %d" %train_cnt)
    print("Size of testset: %d" %test_cnt)
    print("Size of devset: %d" %dev_cnt)
    return X_train, Y_train, X_test, Y_test, X_dev, Y_dev


if __name__=='__main__':

    train = pd.read_csv('./data/train.tsv', sep='\t')
    test = pd.read_csv('./data/test.tsv', sep='\t')
    ngram = Ngram(1)
    X = ngram.get_csr_matrix(train['Phrase'])
    Y = train['Sentiment']
    n_class = Y.value_counts().count()  # 类别数目
    # print(class_num)

    # print(X_train.shape)  # (124848, 16532)
    # print(Y_train.shape)  # (124848, ) 一维矩阵
    # print(X_test.shape)  # (15606, 16532)
    # print(Y_test.shape)  # (15606, )
    if MODE == 'SGD':
        BATCH_SIZE=1

    if classifier == 'logistic':
        l = logistic(lr=LR,epoch=EPOCH,batch_size=BATCH_SIZE,shuffle=SHUFFLE)
        Y = Y.apply(lambda x: 1 if x > 2 else 0).values
        X_train, Y_train, X_test, Y_test, X_dev, Y_dev = split_dataset(X, Y, 0.8, 0.1, True)
        if MODE == 'SGD' or MODE=='mini':
            loss_list,acc_list = l.SGD(X_train,Y_train,X_dev, Y_dev)
        elif MODE == 'BGD':
            loss_list,acc_list = l.BGD(X_train, Y_train, X_dev, Y_dev)
        logistic_plot(loss_list=loss_list,acc_list=acc_list, epoch=EPOCH, mode=MODE)

    elif classifier == 'softmax':
        X_train, Y_train, X_test, Y_test, X_dev, Y_dev = split_dataset(X, Y, 0.8, 0.1, True)
        l = softmax_regression(lr=LR,epoch=EPOCH,batch_size=BATCH_SIZE,shuffle=SHUFFLE,n_class=n_class)
        if MODE == 'SGD' or MODE=='mini':
            acc_list = l.SGD(X_train, Y_train, X_dev, Y_dev)
        elif MODE == 'BGD':
            acc_list = l.BGD(X_train, Y_train, X_dev, Y_dev)
        softmax_plot(acc_list=acc_list, epoch=EPOCH, mode=MODE)

    # predict
    csv_file = open('submit_try.csv','w',encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["PhraseId", "Sentiment"])
    predict_result = l.predict(ngram.get_csr_matrix(test['Phrase'],fix_vocab=True))  # 预测结果 这次不可以根据test集修改矩阵的布局 需要固定vocab
    for id, res in zip(test['PhraseId'],predict_result):
        csv_writer.writerow([id, res])
    csv_file.close()


