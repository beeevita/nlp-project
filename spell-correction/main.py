# coding=utf-8
from load import *
from get_operation_letter import get_operation_letter
from get_candidates import get_candidates_from_trie, get_trie
import difflib
import ast
from ngram import get_gram, Unigram,Bigram,Trigram
from nltk.corpus import reuters,brown
import numpy as np
import ssl
import pickle
ssl._create_default_https_context = ssl._create_unverified_context

def get_channel_model(candidate, error_word):  # confusion_matrix v0.0
    edit_type, edit_distance, x, y = get_operation_letter(error_word, candidate)
    p = 1/V

    # insert  acres acress
    if edit_type[0]:
        x = x["ins"]
        y = y["ins"]
        if x + y in ins_matrix and corpus_str.count(x):
            if x == '#':
                p= (ins_matrix[x+y]+1) / corpus.count(' ' + y)
            else:
                p= ((ins_matrix[x+y])+1) / corpus_str.count(x)
        else:
            p= 1 / len(corpus)
        return np.log(p)

    # delete  acress actress  ct/ct
    if edit_type[1]:
        x = x["del"]
        y = y["del"]
        if x + y in del_matrix and corpus_str.count(x+y):
            p= (del_matrix[x+y]+1 )/ corpus_str.count(x+y)
        elif x + y in del_matrix:
            p= ((del_matrix[x+y]) +1)/ len(corpus)
        elif corpus_str.count(x+y):
            p= 1/corpus_str.count(x+y)
        else:
            p= 1 / len(corpus)
        return np.log(p)

    # sub  w:access  x:acress  rc/c
    if edit_type[2]:
        x = x["sub"]
        y = y["sub"]
        if (x+y)[0:2] in sub_matrix and corpus.count(y):
            p= (sub_matrix[(x + y)[0:2]]+1) / corpus.count(y)
        elif (x+y)[0:2] in sub_matrix:
            p= (sub_matrix[(x+y)[0:2]]+1)/len(corpus)
        elif corpus_str.count(y):
            p= 1/corpus_str.count(y)
        else:
            p= 1 / len(corpus)
        return np.log(p)
    # trans
    if edit_type[3]:
        x = x["trans"]
        y = y["trans"]
        if x + y in trans_matrix and corpus_str.count(x + y):
            p= (trans_matrix[x + y]+1) / corpus_str.count(x + y)
        elif x + y in trans_matrix:
            p= (trans_matrix[x + y] + 1) / len(corpus)
        elif corpus_str.count(x + y):
            p= 1 / corpus_str.count(x + y)
        else:
            p= 1 / len(corpus)
        return np.log(p)

    return np.log(p)

def get_channel_model_no_smoothing(candidate, error_word):
    edit_type, edit_distance, x, y = get_operation_letter(error_word, candidate)

    p = 1 / (V)
    # insertion
    if edit_type[0]:
        x = x["ins"]
        y = y["ins"]
        if(x+y+"|"+ x) in ins_matrix and corpus_str.count(x):
            if x == " ":
                if corpus_str.count(" " + y):
                    p = (ins_matrix[(x+y+"|"+x)]) / (corpus_str.count(" " +y))
                else:
                    p =1/(corpus_str.count(x))
            else:
                p = (ins_matrix[(x+y+"|" +x)])/(corpus_str.count(x))

        elif corpus_str.count(x):
            p = 1 / (corpus_str.count(x) )
        elif (x+y + "|" + x) in ins_matrix:
            p = (ins_matrix[(x+y + "|" +x)]) / (V)
        else:
            p = 1 / (V)
        return np.log(p)

    # deletion
    if edit_type[1]:
        x = x["del"]
        y = y["del"]
        if (x+"|"+ x+y) in del_matrix and corpus_str.count(x + y):
            p = (del_matrix[(x+"|"+x+y)]) / (corpus_str.count(x+y) )
        elif corpus_str.count(x + y):
            p = 1 / (corpus_str.count(x+y))
        elif (x+"|"+ x+y) in del_matrix:
            p = (del_matrix[(x+"|"+ x+y)]) / (V)
        else:
            p = 1 / (V)
        return np.log(p)

    # substitution
    if edit_type[2]:
        x = x["sub"]
        y = y["sub"]
        if (x+"|"+y) in sub_matrix and corpus_str.count(y):
            p = (sub_matrix[(x+"|"+y)] + 1) / (corpus_str.count(y))
        elif corpus_str.count(y):
            p = 1 / (corpus_str.count(y))
        elif (x + "|" + y) in sub_matrix:
            p = (sub_matrix[(x+"|"+y)]) / (V)
        else:
            p = 1 / (V)
        return np.log(p)

    # transposition
    if edit_type[3]:
        x = x["trans"]
        y = y["trans"]
        if (y+x+"|"+ x+y) in trans_matrix and corpus_str.count(x + y):
            # for example caress acress c a ->ac|ca
            p = (trans_matrix[(y+x+"|"+ x+y)]) / (corpus_str.count(x + y))
        elif corpus_str.count(x + y):
            p = 1 / (corpus_str.count(x + y))
        elif (y+x+"|"+ x+y) in trans_matrix:
            p = (trans_matrix[(y+x+"|"+ x+y)] )/(V)
        else:
            p = 1 / (V)

        return np.log(p)

    return np.log(p)

def get_channel_model2(candidate, error_word, corpus, lamb):  # confusion_matrix v1.0
    edit_type, edit_distance, x, y = get_operation_letter(error_word, candidate)
    p = 1 / (V * lamb)
    # insertion
    if edit_type[0]:
        x= x["ins"]
        y = y["ins"]
        if(x+y+"|"+ x) in ins_matrix and corpus_str.count(x):
            if x == " ":
                if corpus_str.count(" " + y):
                    p = (ins_matrix[(x+y+"|"+x)] + 1) / (corpus_str.count(" " +y) + (V*lamb))
                else:
                    p =1/(corpus_str.count(x) + (V*lamb))
            else:
                p = (ins_matrix[(x+y+"|" +x)] + lamb)/(corpus_str.count(x) + (V*lamb))

        elif corpus_str.count(x):
            p = 1 / (corpus_str.count(x) + V* lamb)
        elif (x+y + "|" + x) in ins_matrix:
            p = (ins_matrix[(x+y + "|" +x)] + lamb) / (V*lamb)
        else:
            p = 1 / (V*lamb)
        return np.log(p)

    # deletion
    if edit_type[1]:
        x= x["del"]
        y = y["del"]
        if (x+"|"+ x+y) in del_matrix and corpus_str.count(x + y):
            p = (del_matrix[(x+"|"+x+y)] + lamb) / (corpus_str.count(x+y) + V*lamb)
        elif corpus_str.count(x + y):
            p = 1 / (corpus_str.count(x+y) + (V*lamb))
        elif (x+"|"+ x+y) in del_matrix:
            p = (del_matrix[(x+"|"+ x+y)] + lamb) / (V*lamb)
        else:
            p = 1 / (V*lamb)
        return np.log(p)

    # substitution
    if edit_type[2]:
        x= x["sub"]
        y = y["sub"]
        if (x+"|"+y) in sub_matrix and corpus_str.count(y):
            p = (sub_matrix[(x+"|"+y)] + 1) / (corpus_str.count(y) + (V*lamb))
        elif corpus_str.count(y):
            p = 1 / (corpus_str.count(y) + (V*lamb))
        elif (x + "|" + y) in sub_matrix:
            p = (sub_matrix[(x+"|"+y)] + lamb) / (V*lamb)
        else:
            p = 1 / (V*lamb)
        return np.log(p)

    # transposition
    if edit_type[3]:
        x= x["trans"]
        y = y["trans"]
        if (y+x+"|"+ x+y) in trans_matrix and corpus_str.count(x + y):
            # for example caress acress c a ->ac|ca
            p = (trans_matrix[(y+x+"|"+ x+y)] + lamb ) / (corpus_str.count(x + y) + (V*lamb))
        elif corpus_str.count(x + y):
            p = 1 / (corpus_str.count(x + y) + (V*lamb))
        elif (y+x+"|"+ x+y) in trans_matrix:
            p = (trans_matrix[(y+x+"|"+ x+y)] + lamb)/(V*lamb)
        else:
            p = 1 / (V*lamb)
        return np.log(p)

    return np.log(p)

def kneser_ney(w2, w3, w4, vocab_of_corpus, d):
    cur = w2 + " " + w3
    post = w3 + " " + w4

    vocab_pre_cnt = 0
    for item in vocab_of_corpus:
        item = w2 + " " + item
        if item in gram_dict:
            vocab_pre_cnt += gram_dict[item]

    our_cur_cnt = 0
    if cur in gram_dict:
        our_cur_cnt = gram_dict[cur]

    if vocab_pre_cnt == 0:
        vocab_pre_cnt = 1e-7
    if our_cur_cnt == 0:
        our_cur_cnt = 1e-7

    vocab_cur_cnt = 0
    for item in vocab_of_corpus:
        item = item + " " + w3
        if item in gram_dict:
            vocab_cur_cnt += gram_dict[item]
    lamda = d * our_cur_cnt / vocab_pre_cnt
    pre_p = max(our_cur_cnt - d, 0) / vocab_pre_cnt + lamda * vocab_cur_cnt / bi_size

    vocab_post_cnt = 0
    for item in vocab_of_corpus:
        item = item + " " + w3
        if item in gram_dict:
            vocab_post_cnt += gram_dict[item]
    if vocab_post_cnt == 0:
        vocab_post_cnt = 1e-7

    our_post_cnt = 0
    if post in gram_dict:
        our_post_cnt = gram_dict[post]
    if our_post_cnt == 0:
        our_post_cnt = 1e-7

    vocab_cur_cnt = 0
    for item in vocab_of_corpus:
        item = w3 + " " + item
        if item in gram_dict:
            vocab_cur_cnt += gram_dict[item]

    lamda = d * our_post_cnt / (vocab_post_cnt)
    post_p = max(our_post_cnt - d, 0) / (vocab_post_cnt) + lamda * vocab_cur_cnt / bi_size

    return np.log(pre_p)+np.log(post_p)

def spell_correct(lamb,channel_mode,lm_mode,trie,vocab,n,corpus,V,test_path='testdata.txt'):
    test_set = get_testset(test_path)
    write_file = open('result.txt', 'w')
    test_file = open(test_path, 'r')

    result_list = []
    for line in test_file:
        item = line.split('\t')
        result_list.append(item[0]+'\t'+item[2])

    for cnt,item in enumerate(test_set):
        for word in item[2][1:-1]:
            if word in vocab:
                continue
            candidate_list = list(set(get_candidates_from_trie(vocab=vocab, trie=trie, word=word, edit_distance=1)))
            if(len(candidate_list) == 0):
                candidate_list = list(set(get_candidates_from_trie(vocab=vocab, word=word,edit_distance=2, trie=trie)))
            candi_prob = []
            if(len(candidate_list) == 1):
                result_list[cnt] = result_list[cnt].replace(word, candidate_list[0])
                print("sentence " + str(cnt + 1) + ": " + word + " -> " + candidate_list[0])
                continue
            else:
                pxw = 1/V
                for candidate in candidate_list:
                    if channel_mode == 0:
                        pxw = get_channel_model(error_word=word, candidate=candidate)
                    elif channel_mode == 1:
                        pxw = get_channel_model2(word, candidate, corpus,0.01)
                    elif channel_mode == 2:
                        pxw = get_channel_model_no_smoothing(candidate=candidate, error_word=word)
                    else:
                        print("Error, Please choose channel_mode among {0,1,2}")
                        return

                    error_pos = item[2][1:-1].index(word)  # 拼错单词的位置
                    p=1/V

                    if lm_mode == "add1":
                        # unigram
                        if n == 1:
                            p = Unigram(candidate, gram_dict, V)

                        # bigram
                        elif n == 2:
                            left = item[2][error_pos]+' '+candidate
                            right = candidate + ' '+item[2][error_pos+2]
                            p = Bigram(left, gram_dict, V, lamb=1, pos="left") + Bigram(right, gram_dict, V, lamb=1, pos="right")

                        # trigram
                        elif n == 3:
                            p = Trigram(item[2][error_pos]+" "+candidate+" "+item[2][error_pos+2], get_gram(), V, lamb=lamb)

                        p += pxw

                    elif lm_mode == "addk":
                        if n == 1:
                            p = Unigram(candidate, gram_dict, V)

                        # bigram
                        elif n == 2:
                            left = item[2][error_pos] + ' ' + candidate
                            right = candidate + ' ' + item[2][error_pos + 2]
                            p = Bigram(left, gram_dict, V, lamb=1, pos="left") + Bigram(right, gram_dict, V, lamb=lamb,
                                                                                        pos="right")

                        # trigram
                        elif n == 3:
                            p = Trigram(item[2][error_pos] + " " + candidate + " " + item[2][error_pos + 2], get_gram(),
                                        V, lamb=lamb)

                        p += pxw

                    else:
                        pkn = kneser_ney(item[2][error_pos], candidate, item[2][error_pos+2], vocab_of_corpus=vocab_of_corpus,d=0.75)
                        p = pkn + pxw

                    candi_prob.append(p)
                # if(len(candi_prob) == 0):
                #     result_list[cnt] = result_list[cnt]
                #     continue
                max_index = candi_prob.index(max(candi_prob))  # 最大值的位置
                print("sentence "+str(cnt+1)+ ": "+word + " -> "+candidate_list[max_index])
                result_list[cnt] = result_list[cnt].replace(word, candidate_list[max_index])

        write_file.write(result_list[cnt])

if __name__ == '__main__':

    corpus_download = False
    if(corpus_download):
        nltk.download('reuters')
        nltk.download('punkt')
        nltk.download('brown')

    cate = reuters.categories()
    gram_dict, uni_size, bi_size, tri_size = get_gram()  # frequencies of each 1/2/3-gram
    vocab = get_vocab()  # load from "vocab.txt"
    trie = get_trie(vocab)
    ins_matrix, del_matrix, sub_matrix, trans_matrix = get_confusion_matrix()


    print(cate)
    V, corpus,vocab_of_corpus = get_corpus(cate)
    corpus_str = ' '.join(corpus)
    print("Size of Corpus: ", V)

    # n={1,2,3}
    # channel_mode = {0,1,2} 0: 混淆矩阵V0.0  1:混淆矩阵V1.0  2: no smoothing method
    # Lm_mode = {"kneser", "addk", "add1"}  (language_model)
    spell_correct(n=1, V=V, corpus=corpus, vocab=vocab,trie=trie, channel_mode=1, lm_mode="kneser", lamb=0.01)  # no smoothing


