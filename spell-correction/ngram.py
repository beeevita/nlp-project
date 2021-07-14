#coding=utf-8
from __future__ import division
import numpy as np
import math
from nltk.corpus import reuters,brown
import string
import ast
from collections import Counter
from load import *


def get_gram():
    cate = reuters.categories()
    corpus_raw_text = reuters.sents(categories=cate)
    uni_size = 0  # unigram个数
    bi_size = 0  # bigram个数
    tri_size = 0  # trigram 个数
    gram_dict = {}

    for sents in corpus_raw_text:
        sents = ['<s>'] + sents + ['</s>']

        # remove string.punctuation
        for words in sents[::]:
            if (words in string.punctuation):
                sents.remove(words)

        # 1-gram
        for item in sents:
            if item in gram_dict:
                gram_dict[item] += 1
            else:
                gram_dict[item] = 1
            uni_size += 1

        # bigram
        if len(sents)>=2:
            for i in range(len(sents)-1):
                item = sents[i]+' '+sents[i+1]
                if item in gram_dict:
                    gram_dict[item] += 1
                else:
                    gram_dict[item] = 1
                bi_size += 1

        # trigram
        if len(sents) >= 3:
            for i in range(len(sents)-2):
                item = sents[i]+' '+sents[i+1]+' '+sents[i+2]
                if item in gram_dict:
                    gram_dict[item] += 1
                else:
                    gram_dict[item] = 1
                tri_size += 1

    return gram_dict, uni_size, bi_size, tri_size

def Unigram(word, gram_dict,V):
    if word in gram_dict:
        return np.log((gram_dict[word])/(V))
    else:
        return np.log(1/V)


def Bigram(gram, gram_dict, V, lamb, pos):  # gram:"boost protectionist"
    if pos == "left":
        pre = gram.split(' ')[0]  # pre: "boost"
    else:
        pre = gram.split(' ')[1]
    if gram in gram_dict and pre in gram_dict:
        p = (gram_dict[gram] + lamb) / (gram_dict[pre] + lamb*V)
    elif gram in gram_dict:
        p = (gram_dict[gram] + lamb) / V * lamb
    elif pre in gram_dict:
        p = lamb / (gram_dict[pre] + lamb * V)
    else:
        p = 1 / V
    return np.log(p)

def Trigram(gram, gram_dict, V, lamb):
    pre = gram.split(' ')[0] + " "+gram.split(' ')[1]
    post = gram.split(' ')[1] + " "+gram.split(' ')[2]
    return Bigram(pre, gram_dict, V, lamb, "left") + Bigram(post, gram_dict, V, lamb, "left")


class ngram1():
    def __init__(self, n, sent, corpus,V):  # n == 1/2/3 unigram, bigram, trigram
        self.V, self.corpus = V,corpus
        self.sent = sent
        self.n = n

        self.unigram=self.Unigram()
        if n >= 2:
            self.bigram=self.Bigram()
        if n >= 3:
            self.trigram=self.Trigram()
        return


    def Unigram(self):
        unigram = dict()
        unigram = Counter(corpus)
        return unigram

    def Bigram(self):
        biwords = []
        for index, item in enumerate(corpus):
            if index==len(corpus)-1:
                break
            biwords.append(item+' '+corpus[index+1])
        bigram = dict()
        bigram = Counter(biwords)

        return bigram


    def Trigram(self):
        triwords = []
        for index, item in enumerate(corpus):
            if index==len(corpus)-2:
                break
            triwords.append(item+' '+corpus[index+1]+' '+corpus[index+2])
        trigram = dict()
        trigram = Counter(triwords)
        return trigram

    def cal_prob(self, word, gram):
        # print("size of corpus: ", len(self.corpus))
        if self.n == 1:
            if word in self.unigram:
                return math.log((self.unigram[word])/(self.V))
            else:
                return 1/self.V
        elif self.n == 2:
            if gram in self.bigram and word in self.unigram:
                return math.log((self.bigram[gram]+1)/(self.unigram[word]+self.V))  # add-1 smoothing
            elif gram in self.bigram:
                return math.log((self.bigram[gram] + 1) / self.V)
            elif word in self.unigram:
                return math.log(1 / (self.unigram[word]+self.V))
            else:
                return 1/self.V
        elif self.n == 3:
            if gram in self.trigram and word in self.bigram:
                return math.log((self.trigram[gram]+1)/(self.bigram[word]+self.V))
            elif gram in self.trigram:
                return math.log((self.trigram[gram]+1) / self.V)
            elif word in self.bigram:
                return math.log(1/(self.bigram[word]+self.V))
            else:
                return math.log(1/self.V)

    def cal_sent_prob(self, form='antilog'):
        words = self.sent.lower().split()
        P=0
        if self.n == 1:
            for item in words:
                P += self.cal_prob(item, item)
        elif self.n == 2:
            for i, item in enumerate(words):
                if i == len(words)- 1:
                    break
                P += self.cal_prob(item, item+' '+words[i+1])
        elif self.n == 3:
            for i, item in enumerate(words):
                if i == len(words)- 2:
                    break
                P +=self.cal_prob(item+' '+words[i+1], item+' '+words[i+1]+' '+words[i+2])

        if form == 'log':
            return P
        elif form == 'antilog':
            return math.pow(math.e, P)

if __name__=='__main__':
    sentence = "I love you"
    cate = reuters.categories()
    V,corpus= get_corpus(cate=cate)
    ng = ngram1(n=1, sent=sentence,corpus=corpus, V=V)
    print(ng.cal_sent_prob())
    # gram_dict=get_gram()
    # print(gram_dict)


