from nltk.corpus import reuters, brown
import string
import ast
import nltk

def get_corpus(cate):
    corpus = reuters.sents(categories=cate)
    corpus_list = []
    punc = string.punctuation
    for sents in corpus:
        for words in sents[::]:
            if (words in  punc):
                sents.remove(words)
        corpus_list.extend(sents)

    V = len(list(set(corpus_list)))
    vocab_of_corpus = list(set(corpus_list))
    # print("size of corpus: ", V)
    return V, corpus_list, vocab_of_corpus

# ¼ÓÔØ´Ê»ã±í
def get_vocab(vocab_path='vocab.txt'):
    f = open(vocab_path, 'r')
    vocab_file = f.readlines()
    vocab = set()
    for line in vocab_file:
        line = line.strip()
        vocab.add(line)

    print(len(vocab))
    f.close()
    return vocab

def get_testset(test_path):
    f = open(test_path, 'r')
    punc = string.punctuation
    test_file = f.readlines()
    test_data = []
    for line in test_file:
        elem = line.strip().split('\t')
        sentence = elem[2]
        sentence = nltk.word_tokenize(sentence)
        sentence =  ['<s>'] + sentence + ['</s>']
        for word in sentence[::]:
            # É¾µô±êµã·ûºÅ
            if word in punc:
                sentence.remove(word)
            elem[2] = sentence
        test_data.append(elem)
    f.close()
    return test_data


def get_confusion_matrix(path='./confusion_matrix/'):
    file_list = ['ins_v2.txt', 'del_v2.txt', 'sub_v2.txt', 'trans_v2.txt']
    for f in file_list:
        file = open(path+f, 'r')
        content = file.read()
        if f[0] == 'i':
            ins_matrix = ast.literal_eval(content)
        elif f[0] == 'd':
            del_matrix = ast.literal_eval(content)
        elif f[0] == 's':
            sub_matrix = ast.literal_eval(content)
        else:
            trans_matrix = ast.literal_eval(content)
        file.close()

    return ins_matrix, del_matrix, sub_matrix, trans_matrix


