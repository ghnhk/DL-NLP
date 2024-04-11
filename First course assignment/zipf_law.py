# -*- coding = utf-8 -*-
# @Time : 2024/4/7 16:31
# @Author : 牛华坤
# @File : zipf_law.py
# @Software : PyCharm
import os
import jieba
import matplotlib.pyplot as plt
# 返回文件夹下所有文本文件的路径
def getFilesPath(file_dir):
    files_path = []
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for filename in filenames:
            # 将文件路径分割成文件名和扩展名
            if os.path.splitext(filename)[1] == '.txt':
                files_path.append(os.path.join(dirpath, filename))
    return files_path
# 获得汉字字典
def getCharacterDict(files_path,stop_words):
    cdict = {}
    for path in files_path:
        if path.split('\\')[-1] == 'total.txt':
            continue
        for c in open(path, 'rb').read().decode('ANSI'):
            # 汉字的Unicode编码范围是0x4E00至0x9FA5
            # if (19968 <= ord(c) <= 40869) and (c not in stop_words):
            if (19968 <= ord(c) <= 40869):
                cdict[c] = cdict.get(c, 0) + 1
    return cdict
# 获得词语字典
def getWordDict(files_path,stop_words):
    wdict = {}
    for path in files_path:
        if path.split('\\')[-1] == 'total.txt':
            continue
        with open(path, 'rb') as f:
            content = f.read().decode('ANSI')
            # 去掉无意义词汇
            content = content.replace(r'本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
            content = " ".join(jieba.cut(content))  # 得到具体的词语，cut_all默认为精确模式
            word = [w for w in content.strip().split() if len(w) >= 2 and w not in stop_words]  # split默认将所有的空格删除
            for w in word:
                wdict[w] = wdict.get(w, 0) + 1
    return wdict
def showPlt(dict,mode):
    # 将字典按值排序
    sdict = sorted(dict.items(), key=lambda d: d[1], reverse=True)
    print(len(sdict))
    print(sdict[:100])
    ranks = []
    freqs = []
    for rank, value in enumerate(sdict):
        ranks.append(rank + 1)
        freqs.append(value[1])
        rank += 1
    plt.loglog(ranks, freqs)
    plt.grid(True)
    plt.xlabel('rank', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.ylabel('freqs', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.title('Zipf-Law')
    plt.savefig('./Zipf-Law.png')
    plt.show()
if __name__ == '__main__':
    files_path = getFilesPath(r'D:\研一\深度学习与自然语言处理\第一次作业_齐普夫定律\jyxstxtqj_downcc.com')
    # 读取停用词库
    with open(r'D:\研一\深度学习与自然语言处理\第一次作业_齐普夫定律\cn_stopwords.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
        stop_words = f.read().splitlines()
    # cdict = getCharacterDict(files_path,stop_words)
    wdict = getWordDict(files_path,stop_words)
    # showPlt(cdict,'char')
    showPlt(wdict,'word')