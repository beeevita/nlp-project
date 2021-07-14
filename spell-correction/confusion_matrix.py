# coding=utf-8
import difflib
import json
import os
from pprint import pprint
import Levenshtein

def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    edit = [[i+j for j in range(len2+1)] for i in range(len1+1)]
    # print(edit)

    for i in range(1, len1+1):
        for j in range(1, len2+1):
            dele = edit[i-1][j] + 1  # 删除操作
            insert = edit[i][j-1] + 1  # 插入操作
            substitution = edit[i-1][j-1]  # 替换操作
            if word1[i - 1] != word2[j - 1]:
                substitution += 1
            edit[i][j] = min(dele, insert, substitution)
    return edit[len1][len2]


def get_error_dict(src_path="spell-errors.txt"):  # 拼写错误文件路径(spell-error.txt)

    src = open(src_path,"r")
    src_text = src.readlines()
    src_processed = []  # 处理以后的spell-error
    for line in src_text:
        line = line.lower()
        if '-' in line:
            line = line.replace('-', '#')  # 下面会用到diff库 比较结果会出现-字符 所以这里先将-替换为# 以免干扰
            line = line.replace(' ', '$')  # 空格换成$
            # print(line)
        src_processed.append(line)
    src.close()

    error_dict = {}  # 拼写错误字典
    for line in src_processed:
        line = line.strip()  # 去掉末尾换行符
        line = line.split(':')
        correct = line[0]  # 正确的单词
        mis = line[1].split(',')  # 对应的拼写错误的单词
        error_dict[correct] = mis
    return error_dict

# 根据拼写错误的字典计算混淆矩阵
def cal_matrix(error_dict):
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789*,.?#\'_ '
    ins_matrix = {}
    del_matrix = {}
    sub_matrix = {}
    trans_matrix = {}
    d = difflib.Differ()
    # initial
    for x in chars:
        for y in chars:
            del_matrix[x+y] = 0  # xy typed as x
            ins_matrix[x+y] = 0  # x typed as xy
            sub_matrix[x+y] = 0  # x typed as y
            trans_matrix[x+y] = 0  # xy typed as yx

    for correct in error_dict:
        for error in error_dict[correct]:
            # print(correct, error)
            if(edit_distance(correct, error)) <= 2:
                # print(correct, error)  # raining rainning
                # print(list(d.compare(correct, error)))  # ['  r', '  a', '  i', '  n', '+ n', '  i', '  n', '  g']
                diff = "".join(list(d.compare(correct, error)))  # list->str      str(r  a  i  n+ n  i  n  g)
                # print(diff)
                diff = diff.replace(' ', '')  # 去掉空格  str(rain+ning)
                # print(diff)

                if diff.count('+') == 0:
                    # delete * 1
                    if diff.count('-') == 1:
                        del_pos = diff.find('-')
                        del_matrix[diff[del_pos-1]+diff[del_pos+1]] += 1
                        print(correct, error, "delete operation", diff[del_pos - 1], diff[del_pos + 1])

                    # delete * 2
                    elif diff.count('-') == 2:
                        del_pos1 = diff.find('-')
                        del_pos2 = diff[del_pos1 + 1:].find('-') + del_pos1 + 1  # 第二个-位置
                        del_matrix[diff[del_pos1 - 1] + diff[del_pos1 + 1]] += 1
                        del_matrix[diff[del_pos2 - 1] + diff[del_pos2 + 1]] += 1

                    else:
                        continue

                elif diff.count('+') == 1:
                    add_pos = diff.find('+')  # +位置

                    if(diff.count('-') == 0):
                        # only insert
                        # e.g.: abc/abdc  ab+dc
                        print(correct, error, "insert operation", diff[add_pos-1], diff[add_pos+1])
                        ins_matrix[diff[add_pos-1]+diff[add_pos+1]] += 1

                    elif(diff.count('-') == 1):
                        del_pos = diff.find('-')  # -位置
                        # substitue
                        # -和+挨在一起 说明是substitute e.g.: kage/cage diff: -c+kage
                        if abs(add_pos - del_pos) == 2:
                            print(correct,error,"substitute operation: ",(diff[add_pos+1],diff[del_pos+1]))
                            sub_matrix[(diff[del_pos+1]+diff[add_pos+1])] += 1

                        # trans
                        # e.g.: wednesday wedensday diff: wed+en-esday, ne typed as en
                        elif abs(add_pos - del_pos) == 3 and diff[add_pos +1] == diff[del_pos+1]:
                            print(correct, error, "trans operation: ", (diff[del_pos - 1], diff[del_pos + 1]))
                            trans_matrix[diff[del_pos-1]+diff[del_pos+1]] += 1

                        # insert & delete
                        # e.g.:roars/rorse  diff:ro-ars+
                        else:
                            print("2 operations")
                            # print(diff)
                            print(correct, error, "insert operation", diff[add_pos - 1], diff[add_pos + 1])
                            ins_matrix[diff[add_pos - 1] + diff[add_pos + 1]] += 1
                            print(correct, error, "delete operation", diff[del_pos - 1], diff[del_pos + 1])
                            sub_matrix[(diff[del_pos - 1] + diff[del_pos + 1])] += 1

                    elif diff.count('-') == 2:

                        del_pos1 = diff.find('-')
                        del_pos2 = diff[del_pos1 + 1:].find('-') + del_pos1 + 1  # 第二个-位置

                        # 与del作为trans
                        if abs(add_pos - del_pos1) == 3 and diff[add_pos + 1] == diff[del_pos1 + 1]:
                            print("2 operations")
                            print(correct, error, "trans operation: ", (diff[del_pos1 - 1], diff[del_pos1 + 1]))
                            print(correct, error, "delete operation", (diff[del_pos2 - 1], diff[del_pos2 + 1]))
                            trans_matrix[diff[del_pos1 - 1]+diff[del_pos1 + 1]] += 1

                            del_matrix[diff[del_pos2 - 1]+diff[del_pos2 + 1]] += 1

                        # 与del2作为trans
                        elif abs(add_pos - del_pos2) == 3 and diff[add_pos + 1] == diff[del_pos2 + 1]:
                            print("2 operations")
                            print(correct, error, "trans operation: ", (diff[del_pos2 - 1], diff[del_pos2 + 1]))
                            print(correct, error, "delete operation", (diff[del_pos1 - 1], diff[del_pos1 + 1]))
                            trans_matrix[diff[del_pos2 - 1]+ diff[del_pos2 + 1]] += 1
                            del_matrix[diff[del_pos1 - 1]+diff[del_pos1 + 1]] += 1

                        # 与del1形成sub
                        elif abs(add_pos - del_pos1) == 2 and (del_pos2 - del_pos1) != 2:
                            print("2 operations")
                            print(correct, error, "substitute operation: ", (diff[del_pos1 + 1], diff[add_pos + 1]))
                            sub_matrix[diff[del_pos1 + 1]+diff[add_pos + 1]] += 1
                            print(correct, error, "delete operation", (diff[del_pos2 - 1], diff[del_pos2 + 1]))
                            del_matrix[diff[del_pos2 - 1] + diff[del_pos2 + 1]] += 1

                        elif abs(add_pos - del_pos2) == 2 and (del_pos2 - del_pos1) != 2:
                            print("2 operations")
                            print(correct, error, "substitute operation: ", (diff[del_pos1 + 1], diff[add_pos + 1]))
                            sub_matrix[diff[del_pos1 + 1]+diff[add_pos + 1]] += 1

                            print(correct, error, "delete operation", (diff[del_pos2 - 1], diff[del_pos2 + 1]))
                            del_matrix[diff[del_pos2 - 1] + diff[del_pos2 + 1]] += 1


                elif diff.count('+') == 2:
                    print(diff)
                    add_pos1 = diff.find('+')  # 第一个+位置
                    add_pos2 = diff[add_pos1+1:].find('+') + add_pos1+1  # 第二个+位置

                    # 两个单独的insert操作
                    if diff.count('-') == 0:
                        ins_matrix[diff[add_pos1 - 1] + diff[add_pos1 + 1]] += 1
                        ins_matrix[diff[add_pos2 - 1] + diff[add_pos2 + 1]] += 1
                    elif diff.count('-') == 1:
                        del_pos = diff.find('-')

                        # 与add1构成substitute操作
                        if abs(del_pos - add_pos1) == 2:
                            sub_matrix[(diff[del_pos+1]+diff[add_pos1+1])] += 1
                            ins_matrix[diff[add_pos2 - 1] + diff[add_pos2 + 1]] += 1

                        # 与add2构成substitute操作
                        elif abs(del_pos - add_pos2) == 2:
                            sub_matrix[(diff[del_pos+1]+diff[add_pos2+1])] += 1
                            ins_matrix[diff[add_pos1 - 1] + diff[add_pos1 + 1]] += 1

                        # 与add1构成trans操作
                        elif abs(add_pos1 - del_pos) == 3 and diff[add_pos1 +1] == diff[del_pos+1]:
                            trans_matrix[diff[del_pos-1]+diff[del_pos+1]] += 1
                            ins_matrix[diff[add_pos2 - 1] + diff[add_pos2 + 1]] += 1

                        # 与add2构成trans操作
                        elif abs(add_pos2 - del_pos) == 3 and diff[add_pos2 +1] == diff[del_pos+1]:
                            trans_matrix[diff[del_pos-1]+diff[del_pos+1]] += 1
                            ins_matrix[diff[add_pos1 - 1] + diff[add_pos1 + 1]] += 1

                    # 两次sub或者两次trans操作
                    elif diff.count('-') == 2:
                        del_pos1 = diff.find('-')
                        del_pos2 = diff[del_pos1 + 1:].find('-') + del_pos1 + 1  # 第二个-位置
                        # 两次sub操作
                        if abs(del_pos1 - add_pos1) == 2  and abs(del_pos2 - add_pos2) == 2:
                            sub_matrix[(diff[del_pos1 + 1] + diff[add_pos1 + 1])] += 1
                            sub_matrix[(diff[del_pos2 + 1] + diff[add_pos2 + 1])] += 1

                        # 两次trans操作
                        elif abs(add_pos1 - del_pos1) == 3 and diff[add_pos1 +1] == diff[del_pos1+1] and \
                            abs(add_pos2 - del_pos2) == 3 and diff[add_pos2 + 1] == diff[del_pos2 + 1]:
                            trans_matrix[diff[del_pos1 - 1] + diff[del_pos1 + 1]] += 1
                            trans_matrix[diff[del_pos2 - 1] + diff[del_pos2 + 1]] += 1

                else:
                    continue

            else: # 不考虑编辑距离>2的情况
                continue
    return ins_matrix, del_matrix, sub_matrix, trans_matrix

def cal_matrix_v2(path='count_1edit.txt'):  # get 4 matrixes from the file 'count.1edit.txt'
    f = open(path, "r")
    ins_matrix = {}
    del_matrix = {}
    sub_matrix = {}
    trans_matrix = {}
    for line in f.readlines():
        item = line.strip().split("\t")
        elem = (item[0]).split("|")
        freq = int(item[1])
        
        # insertion
        # ab|a  a typed as ab
        # |前面的是出错的，后面是正确的
        if len(elem[0])==2 and len(elem[1])==1:
                ins_matrix[(elem[0]+"|"+ elem[1])] = freq

        # deletion
        # a|ab: ab typed as a
        if len(elem[0])==1 and len(elem[1])==2:
            del_matrix[(elem[0]+"|"+ elem[1])] = freq

        # sub
        # a|b:  b typed as a
        if len(elem[0])==1 and len(elem[1])==1:
            sub_matrix[(elem[0]+"|"+ elem[1])] = freq

        # trans
        # ab|ba: ba typed as ab
        if len(elem[0])==2 and len(elem[1])==2:
            trans_matrix[(elem[0]+"|"+ elem[1])] = freq

    return ins_matrix, del_matrix, sub_matrix, trans_matrix

# print(edit_distance("abc", "ad"))
# print(Levenshtein.distance("abc", "ad"))

if __name__ == "__main__":
    error_dict = get_error_dict()  # correct -> mistakes
    # matrix = cal_matrix(error_dict)
    matrix = cal_matrix_v2()
    public_path = './confusion_matrix/'
    matrix_filename  = ['ins', 'del', 'sub', 'trans']
    for i, m in enumerate(matrix_filename):
        path = os.path.join(public_path, m)
        # print(path)
        f = open(path+'.txt', 'w')
        f.write(str(matrix[i]))

