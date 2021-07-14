#coding=utf-8
from collections import Counter
import string
import pandas as pd
import os
from scipy.sparse import csr_matrix
import nltk


class Ngram():
    def __init__(self, n, max_tf=0.8):
        self.n = n
        self.token2id = {}
        self.gram_dict = {}
        self.max_tf = max_tf

    def get_gram_tokens(self, tokens):  # ['a', 'series']
        gram_tokens = []
        for i in range(0, len(tokens) - self.n + 1):
            gram_tokens.append(' '.join(tokens[i:i + self.n]))  # 如果n==2,  ['a', 'series', 'of']->['a series', 'series of']
        return gram_tokens

    def get_gram_dict(self,dataset):
        gram_dict = {}
        text = []
        for sentence in dataset:
            sentence = nltk.word_tokenize(sentence.lower())
            text.append(sentence)

        for sents in text:
            if self.n >=1:
                # 1-gram
                for item in sents:
                    if item in gram_dict:
                        gram_dict[item] += 1
                    else:
                        gram_dict[item] = 1

            if self.n >= 2:
                # bigram
                if len(sents) >= 2:
                    for i in range(len(sents) - 1):
                        item = sents[i] + ' ' + sents[i + 1]
                        if item in gram_dict:
                            gram_dict[item] += 1
                        else:
                            gram_dict[item] = 1

            if self.n == 3:
                # trigram
                if len(sents) >= 3:
                    for i in range(len(sents) - 2):
                        item = sents[i] + ' ' + sents[i + 1] + ' ' + sents[i + 2]
                        if item in gram_dict:
                            gram_dict[item] += 1
                        else:
                            gram_dict[item] = 1

        return gram_dict

    def get_csr_matrix(self, data,  fix_vocab=False):
        if not fix_vocab:
            self.gram_dict = self.get_gram_dict(data)
            self.gram_dict = dict(filter(lambda x: x[1] < self.max_tf * len(data), self.gram_dict.items()))
            self.token2id = dict([(k, i) for i, k in enumerate(self.gram_dict.keys())])

        # 构建稀疏矩阵 每个词在每个句子中出现的次数
        # 非零数值有nums[ indptr[i]: indptr[i+1] ]
        # 非零数值的列索引有indices[ indptr[i]: indptr[i+1]]
        indices = []  # 非零值的列索引
        indptr = [0]  # 第i行的的非零数据 nums[i：i+1]
        nums = []  # 所有的非零值
        for line in data:
            tokens = nltk.word_tokenize(line.lower())
            ngrams_counter = Counter(self.get_gram_tokens(tokens))
            for k, v in ngrams_counter.items():
                if k in self.token2id:
                    indices.append(self.token2id[k])
                    nums.append(v)  
            indptr.append(len(indices))
        # print(csr_matrix((nums, indices, indptr), dtype=int, shape=(len(data), len(self.token2id))))
        return csr_matrix((nums, indices, indptr), dtype=int, shape=(len(data), len(self.token2id)))

if __name__== '__main__':
    data_path = 'data'
    train = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t')
    test = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t')

    ngram = Ngram(1)
    gram_dict = ngram.train(train['Phrase'])
    print(gram_dict)

