# -*- coding = utf-8 -*-
# @Time : 2024/5/24 14:58
# @Author : 牛华坤
# @File : pre_pro.py
# @Software : PyCharm
import gensim.models as models
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os

def getWord2VecModel(mode='CBOW'):
    sentences = []
    with open("data/corpus.txt", "r") as f:
        txt = f.read()
        data = txt.splitlines()
        for sentence in data:
            sentence = sentence.strip()
            words = sentence.split(' ')
            sentences.append(words)
        f.close()
    # mode：CBOW,skip-gram
    if(mode == "CBOW"):
        model = models.Word2Vec(sentences=sentences, vector_size=200, window=5, min_count=5, sg=0)
        model.save('w2v_model/CBOW.model')  # 保存模型
    elif(mode == "skip-gram"):
        model = models.Word2Vec(sentences=sentences, vector_size=200, window=5, min_count=5, sg=1)
        model.save('w2v_model/skip-gram.model')  # 保存模型
    else:
        print("模型错误")

def getGloVeModel():
    glove_file = 'data/vectors.txt'
    w2v_file = 'data/embedd.txt'
    # 开始转换
    glove2word2vec(glove_file, w2v_file)
    model = models.KeyedVectors.load_word2vec_format(w2v_file)
    # 将词向量保存，方便下次直接导入
    model.save("gv_model/glove_embedding.model")

def similarity(master, slave, model, mode):
    i = 0
    for m in master:
        print(m, '相似度测试')
        print('-' * 40)
        s = slave[i]
        for c in s:
            if(mode == 'w2v'):
                print(model.wv.similarity(m, c))
            if(mode == 'gv'):
                print(model.similarity(m, c))
        i += 1

def most_similar(master, model, mode):
    for m in master:
        print(m, '最邻近测试')
        print('-' * 40)
        if(mode == 'w2v'):
            print(model.wv.most_similar(m, topn=7))
        if(mode == 'gv'):
            print(model.most_similar(m, topn=7))

def find_relation(a, b, c, model, mode):
    if(mode == 'w2v'):
        d= model.wv.most_similar(positive=[c, b], negative=[a], topn=7)
        print(c,d)
    if(mode == 'gv'):
        d = model.most_similar(positive=[c, b], negative=[a], topn=7)
        print(c, d)

def tsne_plot(model, words_num, mode):
    labels = []
    tokens = []
    terms = []
    vocab = {}
    with open("tools/金庸小说全人物.txt", "r") as f:
        terms += f.read().splitlines()
        f.close()
    if(mode == 'w2v'):
        model = model.wv
    vocab = model.key_to_index

    for word in vocab:
        if(word in terms):
            tokens.append(model[word])
            labels.append(word)
    tokens = np.array(tokens)
    font = FontProperties(fname="himalaya.ttf", size=20)
    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(10, 10))
    for i in range(words_num):
        plt.scatter(x[i], y[i])
        if b'\xe0' in bytes(labels[i], encoding="utf-8"):
            this_font = font
        else:
            this_font = 'SimHei'
        plt.annotate(labels[i],
                     fontproperties=this_font,
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

if __name__ == '__main__':
    # mode：CBOW,skip-gram，先生成corpus.txt文件
    if(not os.access('w2v_model/CBOW.model',os.F_OK)):
        getWord2VecModel("CBOW")
    if(not os.access('w2v_model/skip-gram.model', os.F_OK)):
        getWord2VecModel("skip-gram")
    if(not os.access('gv_model/glove_embedding.model', os.F_OK)):
        getGloVeModel()
    # Word2Vec模型
    # model_path = 'w2v_model/skip-gram.model'
    # model = models.Word2Vec.load(model_path)
    # mode = 'w2v'
    # GolVe模型
    model = models.KeyedVectors.load('gv_model/glove_embedding.model')
    mode = 'gv'

    # master = ['韦小宝','郭靖','杨过','张无忌']
    # slave = [['双儿','阿珂','沐剑屏','方怡','曾柔','建宁公主','苏荃'],
    #          ['黄蓉','洪七公','黄药师','周伯通','梅超风','欧阳克','丘处机'],
    #          ['小龙女','黄蓉','李莫愁','金轮法王','欧阳锋','公孙止','陆无双'],
    #          ['赵敏','张翠山','殷素素','谢逊','周芷若','小昭','成昆']]
    # # 相似度测试
    # similarity(master, slave, model, mode)
    # # 最接近的七个词
    # most_similar(master, model, mode)
    # # 测试相对性
    # find_relation('杨过', '小龙女', '郭靖', model, mode)
    # find_relation('降龙十八掌', '洪七公', '蛤蟆功', model, mode)
    # find_relation('武当派', '张翠山', '丐帮', model, mode)
    # 可视化
    tsne_plot(model, 50, mode)
    print('只求有朝再相逢')