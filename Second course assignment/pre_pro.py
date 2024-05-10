# -*- coding = utf-8 -*-
# @Time : 2024/5/8 15:31
# @Author : 牛华坤
# @File : pre_pro.py
# @Software : PyCharm
import jieba
import jieba.analyse
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import csv

def readFile(filename, is_word=False):
    # 如果未指定名称，则默认为类名
    file_path = "txt/" + filename + ".txt"
    with open(file_path, "r", encoding='gbk', errors='ignore') as f:
        data = f.read()
        data = data.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        f.close()
    # 分词
    with open("tools/cn_stopwords.txt", "r", encoding='utf-8') as f:
        stop_words = f.read().splitlines()
        f.close()
    split_word = []
    if is_word:
        for char in data:
            if (char not in stop_words) and (not char.isspace()):
                split_word.append(char)
    else:
        for word in jieba.cut(data):
            if (word not in stop_words) and (not word.isspace()):
                split_word.append(word)
    return split_word

def getParagraphs(is_word, K):
    # 读取小说名字
    word_len = 0
    with open("txt/inf.txt", "r") as f:
        files_name = f.read().split(',')
        dict = {}
        for name in files_name:
            data = readFile(name, is_word)
            word_len += len(data)
            dict[name] = data
        f.close()
    # 计算每篇文章抽取段落数
    id = 1
    con_list = []
    for name in files_name:
        count = math.floor(len(dict[name])/word_len * 1000 + 0.48)
        print(name,':',count)
        # 特殊处理
        if count == 0:
            count = 1
        pos = int(len(dict[name]) // count)
        data_temp=[]
        for i in range(count):
            start = i * pos
            end = i * pos + K
            if(end >= len(dict[name])):
                end = len(dict[name]) - 1
                start = end - K
            data_temp = data_temp + dict[name][start:end]
            con = {
                'id': id,
                'label': name,
                'data': data_temp
            }
            con_list.append(con)
            id += 1
    if is_word:
        save_path = 'data/chars' + '_' + str(K) + '.csv'
    else:
        save_path = 'data/words' + '_' + str(K) + '.csv'
    with open(save_path, 'a', newline='', encoding='utf-8') as f:
        csv_header = ['id', 'label', 'data']  # 设置表头，即列名
        csv_writer = csv.DictWriter(f, csv_header)
        if f.tell() == 0:
            csv_writer.writeheader()
        csv_writer.writerows(con_list)  # 写入数据

if __name__ == "__main__":
    # 抽取数据集
    K_number = [20,100,500,1000,3000]
    for K in K_number:
        getParagraphs(True,K)
        getParagraphs(False,K)
