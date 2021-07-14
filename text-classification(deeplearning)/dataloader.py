#coding=utf8
import random
import torch
import os
import spacy
from torch.nn import init
import pandas as pd
from torchtext.data import Field,TabularDataset,BucketIterator, Iterator

def load_data(path, train_rate,test_rate,shuffle=True):
    assert train_rate + test_rate <= 1.0 and train_rate + test_rate > 0

    dataset = pd.read_csv(path, sep='\t')
    dataset.drop('PhraseId',axis=1)
    dataset.drop('SentenceId', axis=1)
    total = dataset.shape[0]
    index = [i for i in range(total)]
    if shuffle:
        random.shuffle(index)
    # print(index)

    train_cnt = int(train_rate * total)
    test_cnt = int(test_rate * total)
    dev_cnt = total - train_cnt - test_cnt

    train_set = dataset.iloc[index[:train_cnt],:]
    dev_set = dataset.iloc[index[train_cnt:train_cnt+dev_cnt],:]
    test_set = dataset.iloc[index[:test_cnt],:]
    print(train_set)
    print("Size of trainset: %d" %train_cnt)
    print("Size of testset: %d" %test_cnt)
    print("Size of devset: %d" %dev_cnt)
    train_set.to_csv('./data/split_train.csv', index=False)
    test_set.to_csv("./data/split_test.csv", index=False)
    dev_set.to_csv("./data/split_dev.csv", index=False)


def Dataloader(batch_size,device, embedding_dim):
    PAD_TOKEN = '<pad>'
    TEXT = Field(sequential=True, batch_first=True, lower=True, pad_token=PAD_TOKEN)
    LABEL = Field(sequential=False, batch_first=True, unk_token=None)


    # 读取数据,不需要的filed设置为None
    fields = [("PhraseId", None),("SentenceId", None), ('Phrase', TEXT),('Sentiment', LABEL)]
    train_data = TabularDataset(path='./data/split_train.csv', format='csv',fields=fields, skip_header=True)
    dev_data = TabularDataset(path='./data/split_dev.csv', format='csv', fields=fields,skip_header=True)
    test_data = TabularDataset(path='./data/split_test.csv', format='csv',fields=fields,skip_header=True,)
    if embedding_dim==100:
        vectors = 'glove.6B.100d'
    elif embedding_dim==50:
        vectors = 'glove.6B.50d'
    elif embedding_dim==200:
        vectors = 'glove.6B.200d'
    else:
        vectors = 'glove.6B.300d'

    # 构建词典，字符映射到embedding
    # TEXT.vocab.vectors 就是词向量
    # unk_init:当corpus 中有的 token 在 vectors 中不存在时 的初始化方式
    TEXT.build_vocab(train_data, vectors=vectors, unk_init=lambda x: torch.nn.init.uniform_(x, a=-0.25, b=0.25))
    LABEL.build_vocab(train_data)

    # 得到索引，PAD_TOKEN='<pad>'
    PAD_INDEX = TEXT.vocab.stoi[PAD_TOKEN]
    TEXT.vocab.vectors[PAD_INDEX] = 0.0

    train_iter = BucketIterator(train_data, batch_size=batch_size,train=True, shuffle=True, device=device)
    dev_iter = Iterator(dev_data, batch_size=len(dev_data), train=False, sort=False, device=device)
    test_iter = Iterator(test_data, batch_size=len(test_data), train=False, sort=False, device=device)

    vocab_size=len(TEXT.vocab)
    label_num=len(LABEL.vocab)
    print(label_num)
    # print(LABEL.vocab.freqs)
    print("Size of vocab: %d" %vocab_size)
    return train_iter, dev_iter, test_iter, TEXT.vocab.vectors, LABEL.vocab

if __name__ == '__main__':
    load_data('./data/train.tsv', 0.8, 0.1, shuffle=True)
    # train_iter, dev_iter, test_iter, vectors, label_vocab = Dataloader(16)