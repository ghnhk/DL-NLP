# -*- coding = utf-8 -*-
# @Time : 2024/4/9 17:20
# @Author : 牛华坤
# @File : cal_info_entropy.py
# @Software : PyCharm
import os
import jieba
import math
import csv
basePath = r"D:\研一\深度学习与自然语言处理\第一次作业_齐普夫定律\jyxstxtqj_downcc.com"
infPath = os.path.join(basePath, 'inf.txt')
totalPath = os.path.join(basePath, 'total.txt')
resultPath = './result.csv'
def getFilesPath(file_dir):
    if not os.path.exists(file_dir):
        raise Exception('File not exists')
    with open(file_dir, 'r', encoding='gbk') as f:
        cxt = f.readline()
        files_name = cxt.split(',')  # inf.txt用','隔开文件
        files_path = map(lambda file_name: os.path.join(basePath, file_name + '.txt'), files_name)
    return files_path
def writeTotalFile():
    # 读取inf.txt，得到文件列表
    files_path = getFilesPath(infPath)
    if not os.path.exists(os.path.join(basePath, 'total')):
        os.mkdir(os.path.join(basePath, 'total'))
    totalFile = open(totalPath, 'w', encoding='gb18030')
    for path in files_path:
        with open(path, 'r', encoding='gb18030') as f:
            content = f.read()
            # 写入综合文件，并加入换行
            totalFile.write(content)
            totalFile.write('\n')
            totalFile.flush()
def readFile(filePath, chars, words):
    # 读取停用词库
    with open(r'D:\研一\深度学习与自然语言处理\第一次作业_齐普夫定律\cn_stopwords.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
        stop_words = f.read().splitlines()
    txt = open(filePath, 'r', encoding='ANSI')
    content = txt.read()
    # 去掉无意义词汇
    content = content.replace(r'本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
    for char in content:
        if char not in stop_words and (not char.isspace()):
            chars.append(char)
    # 分词之后，按词存储
    wordl = jieba.lcut(content)
    for word in wordl:
        if word not in stop_words:
            words.append(word)
# 返回一个字/词组字典，格式为 string->int ,即char或word的计数dict
def unigramFrequencyCount(unit):
    cntMap = {}
    if unit is None:
        raise Exception("Wrong mode for frequency Count!")
    for uniTup in unit:
        cntMap[uniTup] = cntMap.get(uniTup, 0) + 1
    return cntMap
# 返回一个二元字/词组字典，格式为 (string, string)->int ,即2个char或word组成元组的计数dict
def bigramFrequencyCount(unit):
    cntMap = {}
    # bigram要求序列必须大于等于2个符号
    if unit is None or len(unit) <= 1:
        raise Exception("Error in bigram frequency Count!")
    for i in range(len(unit) - 1):
        biTup = (unit[i], unit[i + 1])
        cntMap[biTup] = cntMap.get(biTup, 0) + 1
    return cntMap
# 返回一个三元字/词组字典，格式为 (string, string, string)->int ,即3个char或word组成元组的计数dict
def trigramFrequencyCount(unit):
    cntMap = {}
    # trigram要求序列必须大于等于3个符号
    if unit is None or len(unit) <= 2:
        raise Exception("Error in trigram frequency Count!")
    for i in range(len(unit) - 2):
        triTup = (unit[i], unit[i + 1], unit[i + 2])
        cntMap[triTup] = cntMap.get(triTup, 0) + 1
    return cntMap
# 返回unigram模型计算的平均信息熵
def unigramEntropy(unit, name, mode):
    unigramFreqMap = unigramFrequencyCount(unit)
    length = len(unit)
    entropy = 0
    for item in unigramFreqMap.items():
        pr = item[1] / length
        entropy += -1.0 * pr * math.log(pr, 2)
    print("For {}, Unigram Entropy in mode {} is: {}".format(name, mode, entropy))
    return entropy
# 返回bigram模型计算的平均信息熵
def bigramEntropy(unit, name, mode):
    unigramFreqMap = unigramFrequencyCount(unit)
    bigramFreqMap = bigramFrequencyCount(unit)
    length = len(unit) - 1  # 2字/词序列共有len-1个
    entropy = 0
    for item in bigramFreqMap.items():
        prUnion = item[1] / length  # 联合概率，即P(x, y)
        prCond = prUnion / (unigramFreqMap[item[0][0]] / (length + 1))  # 条件概率，
        # 即P(x|y)=P(x, y)/P(y)，注意x是后一个元素，y是前一个
        entropy += -1.0 * prUnion * math.log(prCond, 2)
    print("For {}, Bigram Entropy in mode {} is: {}".format(name, mode, entropy))
    return entropy
# 返回trigram模型计算的平均信息熵
def trigramEntropy(unit, name, mode):
    bigramFreqMap = bigramFrequencyCount(unit)
    trigramFreqMap = trigramFrequencyCount(unit)
    length = len(unit) - 2  # 2字/词序列共有len-2个
    entropy = 0
    for item in trigramFreqMap.items():
        prUnion = item[1] / length  # 联合概率，即P(x, y, z)
        prCond = prUnion / (bigramFreqMap[(item[0][0], item[0][1])] / (length + 1))  # 条件概率，
        # 即P(x|y, z)=P(x, y, z)/P(y, z)，注意x是后一个元素，y, z是前面两个
        entropy += -1.0 * prUnion * math.log(prCond, 2)
    print("For {}, Trigram Entropy in mode {} is: {}".format(name, mode, entropy))
    return entropy
if __name__ == '__main__':
    files_path = list(getFilesPath(infPath))
    files_path.append(totalPath)
    if not os.path.exists(totalPath):
        writeTotalFile()
    # 便于结果写入csv文件，提前定义好csv头
    csvHeader = ['FileName', 'Char-Unigram', 'Word-Unigram', 'Char-Bigram', 'Word-Bigram', 'Char-Trigram', 'Word-Trigram']
    chars = []
    words = []
    entropyMaps = []
    for path in files_path:
        chars.clear()
        words.clear()
        readFile(path,chars,words)
        name = path.split('\\')[-1].replace('.txt','')
        map = {
            'FileName': name,
            'Char-Unigram': unigramEntropy(chars, name, 'char'),
            'Word-Unigram': unigramEntropy(words, name, 'word'),
            'Char-Bigram': bigramEntropy(chars, name, 'char'),
            'Word-Bigram': bigramEntropy(words, name, 'word'),
            'Char-Trigram': trigramEntropy(chars,name,'char'),
            'Word-Trigram': trigramEntropy(words,name,'word')
        }
        entropyMaps.append(map)
    # 计算结束，将结果写入csv文件
    with open(resultPath, 'w', encoding='gbk', newline='') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=csvHeader)
        writer.writeheader()
        writer.writerows(entropyMaps)