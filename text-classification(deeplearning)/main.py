#coding=utf8
import tensorflow as tf
import torch.nn as nn
import torch
import time
from models import CNN, RNN
from dataloader import Dataloader
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd

EMBEDDING = 'glove'  # random
MODEL = 'CNN'
BATCH_SIZE = 128
EPOCH=30
LR = 0.001
N_CLASS= 5
RNN_TYPE = 'CNN'  #  gru, lstm, rnn
cpu=False   #True   False
DROPOUT=0.3
EMBEDDING_DIM=50
max_train_acc = 0
max_test_acc = 0

if cpu :
    USE_CUDA = False
    DEVICE = torch.device('cpu')
else:
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
tf.gfile.DeleteRecursively('./logs')  # 删除之前的记录


if __name__ == '__main__':
    train_iter, dev_iter, test_iter, vectors, label_vocab = Dataloader(BATCH_SIZE, DEVICE, embedding_dim=EMBEDDING_DIM)
    # label_vocab == n_class  种类数
    if MODEL == 'CNN':
        model = CNN(embedding_type=EMBEDDING, pretrained_vectors=vectors,vocab_size=len(vectors), n_class=N_CLASS, dropout=DROPOUT, embedding_dim=EMBEDDING_DIM)
    else:
        model = RNN(vocab_size=len(vectors),embedding_type=EMBEDDING,  n_class=N_CLASS, pretrained_vectors=vectors,rnn_type='rnn', USE_CUDA=USE_CUDA)

    writer = SummaryWriter('logs', comment=MODEL)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 创建优化器SGD
    criterion = nn.CrossEntropyLoss()  # 损失函数
    if USE_CUDA:
        model.cuda()
    best_acc = 0.0
    max_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCH+1):
        model.train()
        total_loss = 0.0
        # 训练
        for i,batch in enumerate(train_iter):
            optimizer.zero_grad()  # 梯度缓存清零
            batch_X = batch.Phrase
            batch_Y = batch.Sentiment
            out = model(batch_X)  # [batch_size, label_num]
            loss = criterion(out, batch_Y)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            batches_done = epoch * len(train_iter) + i  # len(train_iter): 所有的batch个数
            if (i % 100 == 0):
                writer.add_scalar('train_loss', loss, batches_done)
                print("Epoch %d, Step %d, Loss %.2f, average loss: %f" % (epoch, i, loss.item(), total_loss / (i+1)))

        model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total_data_num = len(dev_iter.dataset)

        for i,batch in enumerate(dev_iter):
            batch_X = batch.Phrase
            batch_Y = batch.Sentiment
            out = model(batch_X)
            loss = criterion(out, batch_Y)
            total_loss += loss.item()

            correct = (torch.max(out, dim=1)[1].view(batch_Y.size()) == batch_Y).sum()
            total_correct+=correct.item()

        val_acc = total_correct/ total_data_num
        print("Epoch %d :  val_average Loss: %f, var_acc: %f,Total Time:%f"
              % (epoch, total_loss / (i+1), val_acc, time.time() - start_time))
        writer.add_scalar('val_acc', val_acc, epoch)

        if max_acc < val_acc:
            max_acc = val_acc
            max_test_acc = val_acc
            torch.save(model, 'weights/best')
            print('Model is saved in weights/best')
            # torch.cuda.empty_cache()
        total_loss = 0.0
        total_correct = 0.0
        total_data_num = len(train_iter.dataset)
        for i,batch in enumerate(train_iter):
            batch_X = batch.Phrase
            batch_Y = batch.Sentiment
            out = model(batch_X)
            loss = criterion(out, batch_Y)
            total_loss += loss.item()

            correct = (torch.max(out, dim=1)[1].view(batch_Y.size()) == batch_Y).sum()
            total_correct+=correct.item()

        train_acc = total_correct / total_data_num
        print("Epoch %d :  train_average Loss: %f, train_acc: %f,Total Time:%f"
              % (epoch, total_loss/(i+1), train_acc, time.time() - start_time))
        writer.add_scalar('train_acc', train_acc, epoch)
        if(max_train_acc < train_acc):
            max_train_acc = train_acc

    print('max train accuracy: ', max_train_acc)
    print('max test accuracy: ', max_test_acc)