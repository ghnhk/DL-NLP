# -*- coding = utf-8 -*-
# @Time : 2024/5/24 14:59
# @Author : 牛华坤
# @File : pre_pro.py
# @Software : PyCharm
import jieba
import warnings
import os
warnings.filterwarnings('ignore')

def readFile(filename,stop_words,all_character):
    file_path = "txt/" + filename + ".txt"
    with open(file_path, "r", encoding='gbk', errors='ignore') as f:
        data = f.read()
        data = data.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        data = data.splitlines()
        f.close()
    print("Waiting for {}...".format(filename))
    row_count = 0
    data_seg = []
    for line in data:
        line = line.strip()
        split_word_str = ''
        split_word = jieba.cut(line, cut_all=False)
        for word in split_word:
            if word not in stop_words:
                if word != '\t':
                    if word[:2] in all_character:
                        word = word[:2]
                    split_word_str += word
                    split_word_str += " "
        if len(str(split_word_str.strip())) != 0:
            data_seg.append(str(split_word_str.strip()).split())
    print("{} finished，with {} Row".format(filename, len(data_seg)))
    print("-" * 40)
    return data_seg

def saveCorpus():
    jinyong_termlist = ['金庸小说全人物','金庸小说全武功','金庸小说全门派']
    for termlist in jinyong_termlist:
        with open("tools/"+termlist+".txt", "r") as f:
            terms = f.read().splitlines()
            f.close()
        for term in terms:
            jieba.add_word(term)
    with open("tools/cn_stopwords.txt", "r", encoding='utf-8') as f:
        stop_words = f.read().splitlines()
        f.close()
    with open("tools/金庸小说全人物.txt", "r", encoding='ANSI') as f:
        all_character = f.read().splitlines()
        f.close()
    data = []
    with open("txt/inf.txt", "r") as f:
        files_name = f.read().split(',')
        for name in files_name:
            data += (readFile(name,stop_words,all_character))
        print("-" * 40)
        print("-" * 40)
        print("All finished，with {} Row".format(len(data)))
        f.close()
    with open("data/corpus.txt", "w+") as f:
        for sentence in data:
            for word in sentence:
                f.write(word+' ')
            f.write('\n')
        f.close()

if __name__ == "__main__":
    # 生成corpus.txt文件
    if(not os.access('data/corpus.txt', os.F_OK)):
        saveCorpus()
    print('将相思寄予明月')