# coding=utf-8
from collections import deque
from load import get_vocab

alphabet = "abcdefghijklmnopqrstuvwxyz#$ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 井号代表-，$代表空格
END = '/'

# 生成四种candidates
# insert类型
def get_insertion_candidates(word, vocab):
    # 分割单词 便于insert操作
    split = []
    insertion = []
    # 在word的不同位置插入字母
    for i in range(len(word) + 1):
        split.append(tuple((word[:i], word[i:])))

    for letter in alphabet:
        for left, right in split:
            new_word = left+letter+right
            if new_word in vocab:
                insertion.append(left+letter+right)

    return insertion
# 生成delete类型
def get_del_candidates(word, vocab):
    delete = []
    for i in range(len(word)):
        new_word = word[:i]+word[i+1:]
        if new_word in vocab:
            delete.append(word[:i] + word[1 + i:])
        # print(word[:i]+word[i+1:])

    return delete

# 生成substitute类型
def get_sub_candidates(word, vocab):
    substitute = []
    for i in range(len(word)):
        for letter in alphabet:
            new_word = word[:i]+letter+word[i+1:]
            if new_word in vocab:
                substitute.append(new_word)

    return substitute

# 生成trans类型
def get_trans_candidates(word, vocab):
    trans = []
    for i in range(len(word)-1):
        left = word[:i]
        right = word[i+2:]
        new_word = left + word[i+1] + word[i] + right
        if new_word in vocab:
            trans.append(new_word)
    return trans

def get_all_candidates(word, vocab):
    candidates = []
    candidates = get_del_candidates(word, vocab) + get_sub_candidates(word=word, vocab=vocab) + get_trans_candidates(word, vocab) + get_insertion_candidates(word, vocab)


def get_trie(vocab):
    trie = {}
    for word in vocab:
        t = trie
        for c in word:
            if c not in t:
                t[c] = {}
            t = t[c]
        t[END] = {}
    # print(trie)
    return trie

def get_candidates_from_trie(vocab,word, trie,edit_distance=1):

    que = deque([(trie, word, '', edit_distance)])

    while que:
        trie, word, path, edit_distance = que.popleft()

        if word == '':
            if END in trie:
                yield path
            if edit_distance > 0:
                for k in trie:
                    if k != END:
                        que.appendleft((trie[k], '', path+k, edit_distance-1))
        else:
            if word[0] in trie:
                que.appendleft((trie[word[0]], word[1:], path+word[0], edit_distance))
            if edit_distance > 0:
                edit_distance -= 1
                for k in trie.keys() - {word[0], END}:
                    que.append((trie[k], word[1:], path+k, edit_distance))
                    que.append((trie[k], word, path+k, edit_distance))

                que.append((trie, word[1:], path, edit_distance))

                if len(word) > 1:
                    que.append((trie, word[1]+word[0]+word[2:], path, edit_distance))


if __name__=='__main__':
    vocab = get_vocab()
    trie=get_trie(vocab)
    # print(trie)
    print(list(get_candidates_from_trie(word="acress", trie=trie, vocab=vocab)))



