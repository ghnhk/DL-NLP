# -*- coding = utf-8 -*-
# @Time : 2024/4/7 16:31
# @Author : 牛华坤
# @File : lda_model.py
# @Software : PyCharm
import csv
import pandas as pd
from gensim import corpora
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pyLDAvis.gensim_models
import pyLDAvis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

csv.field_size_limit(500 * 1024 * 1024)

def lda(is_word, K, T):
    if is_word is False:
        file_dir = "data/chars"
        save_dir = "lda_model/chars"

    else:
        file_dir = "data/words"
        save_dir = "lda_model/words"

    filePath = file_dir + "_" + str(K) + ".csv"
    data = pd.read_csv(filePath)

    # 一个主题可以由词汇分布表示，一个段落可以由主题分布表示
    # 在所有段落上建模
    dataset = pd.DataFrame()
    dataset['parag'] = data.iloc[:, -1]
    dataset['parag'] = dataset['parag'].apply(lambda x: eval(x))
    dataList = dataset['parag'].tolist()

    # 构建词典，语料向量化表示
    dict = corpora.Dictionary(dataList)

    # 删掉只在不超过20个文本中出现过的词，删掉在90%及以上的文本都出现了的词
    dict.filter_extremes(no_below=20, no_above=0.9)
    print(dict)
    dict.compactify()  # 去掉因删除词汇而出现的空白

    corpus = [dict.doc2bow(text) for text in dataList]  # 表示为第几个单词出现了几次

    # LDA模型
    print(f'is_word: {is_word}   K:{K}  T:{T}  ----------------')
    ldamodel = LdaModel(corpus, num_topics=T, id2word=dict, passes=40, random_state=6)  # 分为T个主题
    # 字：passes=40, random_state=6

    # 模型评估
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=dataList, dictionary=dict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    per = ldamodel.log_perplexity(corpus)
    print(f'perplexity:{per}     coherence:{coherence_lda}')

    # 提取主题
    topics = ldamodel.show_topics(num_words=T)
    # 输出主题
    for topic in topics:
        print(topic)

    # 保存模型 -----------------------------------------------------------------
    savePath = save_dir + "_" + str(K) + "_" + str(T)
    # dict.save(savePath + "_dict.dict")
    # corpora.MmCorpus.serialize(savePath + "_corpus.mm", corpus)
    ldamodel.save(savePath + "_ldaModel.model")

def lda_test(is_word, K, T):
    if is_word is False:
        file_dir = "data/chars"
        save_dir = "lda_model/chars"
    else:
        file_dir = "data/words"
        save_dir = "lda_model/words"
    filePath = file_dir + "_" + str(K) + ".csv"
    data = pd.read_csv(filePath)

    dataset = pd.DataFrame()
    dataset['parag'] = data.iloc[:, -1]
    dataset['parag'] = dataset['parag'].apply(lambda x: eval(x))
    dataList = dataset['parag'].tolist()

    # 构建词典，语料向量化表示
    dict = corpora.Dictionary(dataList)

    dict.filter_extremes(no_below=20, no_above=0.9)  # 删掉只在不超过20个文本中出现过的词，删掉在90%及以上的文本都出现了的词
    dict.compactify()  # 去掉因删除词汇而出现的空白
    print(dict)
    corpus = [dict.doc2bow(text) for text in dataList]  # 表示为第几个单词出现了几次

    # 加载模型
    savePath = save_dir + "_" + str(K) + "_" + str(T)
    # 加载 LDA 模型
    lda = LdaModel.load(savePath + "_ldaModel.model")

    # 评估模型性能
    coherence_model_lda = CoherenceModel(model=lda, texts=dataList, dictionary=dict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    per = lda.log_perplexity(corpus)
    print(f'perplexity:{per}     coherence:{coherence_lda}')

    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dict, mds='mmds')
    pyLDAvis.show(vis, local=False)


def classify(is_word, K, T):
    if is_word is False:
        file_dir = "data/chars"
        save_dir = "lda_model/chars"
    else:
        file_dir = "data/words"
        save_dir = "lda_model/words"

    filePath = file_dir + "_" + str(K) + ".csv"
    data = pd.read_csv(filePath)

    dataset = pd.DataFrame()
    dataset['parag'] = data.iloc[:, -1]
    dataset['parag'] = dataset['parag'].apply(lambda x: eval(x))
    dataList = dataset['parag'].tolist()

    # 构建词典，语料向量化表示
    dict = corpora.Dictionary(dataList)
    dict.filter_extremes(no_below=20, no_above=0.9)  # 删掉只在不超过20个文本中出现过的词，删掉在90%及以上的文本都出现了的词
    dict.compactify()  # 去掉因删除词汇而出现的空白

    # 加载模型
    savePath = save_dir + "_" + str(K) + "_" + str(T)
    # 加载 LDA 模型
    lda = LdaModel.load(savePath + "_ldaModel.model")

    # 将每个段落进行做主题分布
    topic_matrix = []
    for tmp in dataList:
        cor = dict.doc2bow(tmp)

        topic_distribution = lda.get_document_topics(cor,  minimum_probability=0)
        topic_distribution = [prob for topic, prob in topic_distribution]

        topic_matrix.append(topic_distribution)

    topic_matrix = np.array(topic_matrix)

    # 获取标签
    label = data.iloc[:, -2].tolist()
    label = np.array(label)

    label_encoder = LabelEncoder()
    label_num = label_encoder.fit_transform(label)

    # 训练集训练分类模型
    # 将标签和topic对应，然后划分数据集，进行分类
    result = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(topic_matrix, label_num, test_size=100, random_state=i*20)
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=0)
        # 字：n_estimators=100, max_depth=12, random_state=0
        clf.fit(X_train, y_train)

        x_pred = clf.predict(X_train)
        x_acc = accuracy_score(y_train, x_pred)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        F1 = 2*precision*recall/(precision+recall)
        tmp_result = [accuracy, precision, recall, F1]

        print("-----------------------------------------")
        print(f"{i}:训练集准确率：{x_acc}")
        print(f"{i}:测试集评价：{tmp_result}")

        result.append(tmp_result)

    result = np.array(result)
    print(np.mean(result, axis=0))


if __name__ == '__main__':
    is_word = True

    # 抽取数据集，一共抽取1000个段落，每个段落 K 个token（20,100,500,1000,3000）
    # 段落的标签是对应小说
    K = 500

    # LDA 文本建模，主体数量为T（16，50，100）
    T = 50

    # lda(is_word, K, T)
    # lda_test(is_word, K, T)
    #
    # # 根据主体分布进行分类
    # # 10 次交叉验证
    # classify(is_word, K, T)
