# coding=utf-8
import difflib

# 查看word和对应的candidate之间是哪两个字符有变换 便于应用混淆矩阵
def get_operation_letter(word, candidate):
    if "-" in word:
        word = word.replace("-", "#")
    if "-" in candidate:
        candidate = candidate.replace("-", "#")

    edit_type = [0] * 4  # 代表insertion, delete, substitute, trans的操作次数
    edit_distance = 0
    x = {"ins":"", "del":"", "sub":"", "trans":""}
    y = {"ins":"", "del":"", "sub":"", "trans":""}

    d = difflib.Differ()
    diff = "".join(list(d.compare(candidate, word)))
    diff = diff.replace(' ', '')  # 去掉空格  str(rain+ning)

    if diff.count('+') == 0:
        # delete * 1
        if diff.count('-') == 1:
            del_pos = diff.find('-')
            # print(candidate, word, "delete operation", diff[del_pos - 1], diff[del_pos + 1])
            edit_type[1] = 1  # 1次delete操作
            edit_distance = 1
            x["del"] = diff[del_pos-1]
            y["del"] = diff[del_pos+1]

        # delete * 2
        elif diff.count('-') == 2:
            del_pos1 = diff.find('-')
            del_pos2 = diff[del_pos1 + 1:].find('-') + del_pos1 + 1  # 第二个-位置
            edit_type[1] = 2  # 2次delete操作
            edit_distance = 2
            x["del"] = diff[del_pos1-1]+diff[del_pos2-1]
            y["del"] = diff[del_pos1+1]+diff[del_pos2+1]

    elif diff.count('+') == 1:
        add_pos = diff.find('+')  # +位置

        # insert * 1
        if (diff.count('-') == 0):
            x["ins"],y["ins"] = diff[add_pos-1], diff[add_pos+1]
            edit_type[0] = 1
            edit_distance = 1

        elif (diff.count('-') == 1):
            del_pos = diff.find('-')  # -位置
            # substitue * 1
            if abs(add_pos - del_pos) == 2:
                # print(candidate, word, "substitute operation: ", (diff[del_pos + 1], diff[add_pos + 1]))
                x["sub"] = diff[del_pos + 1]
                y["sub"] = diff[add_pos + 1]
                edit_type[2] = 1
                edit_distance = 1

            # trans * 1
            elif abs(add_pos - del_pos) == 3 and diff[add_pos + 1] == diff[del_pos + 1]:
                # print(diff)
                # print(candidate, word, "trans operation: ", (diff[del_pos - 1], diff[del_pos + 1]))
                x["trans"] = diff[del_pos - 1]
                y["trans"] = diff[del_pos + 1]

                edit_type[3] = 1
                edit_distance = 1

            # insert * 1 & delete * 1
            else:
                # print("2 operations")
                # print(diff)
                # print(candidate, word,  "insert operation", diff[add_pos - 1], diff[add_pos + 1])
                x["ins"] = diff[add_pos - 1]
                y["ins"] = diff[add_pos + 1]
                # print(candidate, word, "delete operation", diff[del_pos - 1], diff[del_pos + 1])

                x["del"] = diff[del_pos - 1]
                y["del"] = diff[del_pos + 1]
                edit_distance = 2
                edit_type[0] = 1
                edit_type[1] = 1

        # del + trans  /  del + sub
        elif diff.count('-') == 2:
            del_pos1 = diff.find('-')
            del_pos2 = diff[del_pos1 + 1:].find('-') + del_pos1 + 1  # 第二个-位置

            # 与第一个del作为trans
            if abs(add_pos - del_pos1) == 3 and diff[add_pos + 1] == diff[del_pos1 + 1]:
                # print("2 operations")
                # print(candidate, word, "trans operation: ", (diff[del_pos1 - 1], diff[del_pos1 + 1]))
                # print(candidate, word, "delete operation", (diff[del_pos2 - 1], diff[del_pos2 + 1]))
                x["trans"] = diff[del_pos1 - 1]
                y["trans"] = diff[del_pos1 + 1]
                edit_type[3] = 1

                x["del"] = diff[del_pos2 - 1]
                y["del"] = diff[del_pos2 + 1]
                edit_type[1] = 1

                edit_distance = 2
            elif abs(add_pos - del_pos2) == 3 and diff[add_pos + 1] == diff[del_pos2 + 1]:
                # print("2 operations")
                # print(candidate, word, "trans operation: ", (diff[del_pos2 - 1], diff[del_pos2 + 1]))
                # print(candidate, word, "delete operation", (diff[del_pos1 - 1], diff[del_pos1 + 1]))
                x["trans"] = diff[del_pos2 - 1]
                y["trans"] = diff[del_pos2 + 1]
                edit_type[3] = 1

                x["del"] = diff[del_pos1 - 1]
                y["del"] = diff[del_pos1 + 1]
                edit_type[1] = 1

                edit_distance = 2

            # del + sub
            # 与del1形成sub
            elif abs(add_pos - del_pos1) == 2 and (del_pos2 - del_pos1) != 2:
                # print("2 operations")
                # print(candidate, word, "substitute operation: ", (diff[del_pos1 + 1], diff[add_pos + 1]))
                x["sub"] = diff[del_pos1 + 1]
                y["sub"] = diff[add_pos + 1]
                edit_type[2] = 1

                # print(candidate, word, "delete operation", (diff[del_pos2 - 1], diff[del_pos2 + 1]))
                x["del"] = diff[del_pos2 - 1]
                y["del"] = diff[del_pos2 + 1]
                edit_type[1] = 1

                edit_distance = 2

            # 与del2形成sub
            elif abs(add_pos - del_pos2) == 2 and (del_pos2 - del_pos1) != 2:
                # print("2 operations")
                # print(candidate, word, "substitute operation: ", (diff[del_pos2 + 1], diff[add_pos + 1]))
                x["sub"] = diff[del_pos2 + 1]
                y["sub"] = diff[add_pos + 1]
                edit_type[2] = 1

                # print(candidate, word, "delete operation", (diff[del_pos1 - 1], diff[del_pos1 + 1]))
                x["del"] = diff[del_pos1 - 1]
                y["del"] = diff[del_pos1 + 1]
                edit_type[1] = 1

                edit_distance = 2


    elif diff.count('+') == 2:
        # print(diff)
        add_pos1 = diff.find('+')  # 第一个+位置
        add_pos2 = diff[add_pos1 + 1:].find('+') + add_pos1 + 1  # 第二个+位置

        # insert * 2
        if diff.count('-') == 0:
            x["ins"] = diff[add_pos1 - 1] + diff[add_pos2 - 1]
            y["ins"] = diff[add_pos1 + 1] + diff[add_pos2 + 1]
            edit_type[0] = 2
            edit_distance = 2

        elif diff.count('-') == 1:
            del_pos = diff.find('-')

            # 与add1构成substitute操作  ins+sub
            if abs(del_pos - add_pos1) == 2:
                x["ins"] = diff[add_pos2 - 1]
                y["ins"] = diff[add_pos2 + 1]
                x["sub"] = diff[del_pos + 1]
                y["sub"] = diff[add_pos1 + 1]
                edit_type[0] = 1  # ins
                edit_type[2] = 1  # sub
                edit_distance = 2

            # 与add2构成substitute操作  ins+sub
            elif abs(del_pos - add_pos2) == 2:
                x["ins"] = diff[add_pos1 - 1]
                y["ins"] = diff[add_pos1 + 1]
                x["sub"] = diff[del_pos + 1]
                y["sub"] = diff[add_pos2 + 1]
                edit_type[0] = 1  # ins
                edit_type[2] = 1  # sub
                edit_distance = 2


            # 与add1构成trans操作  ins+trans
            elif abs(add_pos1 - del_pos) == 3 and diff[add_pos1 + 1] == diff[del_pos + 1]:
                x["ins"] = diff[add_pos2 - 1]
                y["ins"] = diff[add_pos2 + 1]
                x["trans"] = diff[del_pos - 1]
                y["trans"] = diff[del_pos + 1]
                edit_type[0] = 1  # ins
                edit_type[3] = 1  # trans
                edit_distance = 2

            # 与add2构成trans操作  add+trans
            elif abs(add_pos2 - del_pos) == 3 and diff[add_pos2 + 1] == diff[del_pos + 1]:
                x["ins"] = diff[add_pos1 - 1]
                y["ins"] = diff[add_pos1 + 1]
                x["trans"] = diff[del_pos - 1]
                y["trans"] = diff[del_pos + 1]
                edit_type[0] = 1  # ins
                edit_type[3] = 1  # trans
                edit_distance = 2

        # 两次sub或者两次trans操作
        elif diff.count('-') == 2:
            del_pos1 = diff.find('-')
            del_pos2 = diff[del_pos1 + 1:].find('-') + del_pos1 + 1  # 第二个-位置
            # 两次sub操作
            if abs(del_pos1 - add_pos1) == 2 and abs(del_pos2 - add_pos2) == 2:
                x["sub"] = diff[del_pos1 + 1] + diff[del_pos2 + 1]
                y["sub"] = diff[add_pos1 + 1] + diff[add_pos2 + 1]
                edit_type[2] = 2
                edit_distance = 2

            # 两次trans操作
            elif abs(add_pos1 - del_pos1) == 3 and diff[add_pos1 + 1] == diff[del_pos1 + 1] and \
                    abs(add_pos2 - del_pos2) == 3 and diff[add_pos2 + 1] == diff[del_pos2 + 1]:
                x["trans"] = diff[del_pos1 - 1] + diff[del_pos2 - 1]
                y["trans"] = diff[del_pos1 + 1] + diff[del_pos2 + 1]
                edit_type[3] = 2
                edit_distance = 2

    assert sum(edit_type) == edit_distance

    return edit_type, edit_distance, x, y

if __name__=='__main__':
    print(get_operation_letter(word="acress", candidate="caress"))