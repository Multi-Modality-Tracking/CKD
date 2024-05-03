
import os

def where_seq_already(result_path, prefix=""):
    """ 检测哪些序列已经测好了 """
    file_list = os.listdir(result_path)
    seq_list = []
    for item in file_list:
        if item[-8:] == "time.txt":
            seq_list.append(item.split('.')[0][len(prefix):-5])
    return seq_list, len(seq_list)


def seqs_intersect(seq_lists):
    """求交集"""
    seq_list = []
    for seq in seq_lists[0]:
        for i in range(1, len(seq_lists)):
            if seq not in seq_lists[i]:
                continue
            elif i==(len(seq_lists)-1):
                seq_list.append(seq)
    return seq_list, len(seq_list)