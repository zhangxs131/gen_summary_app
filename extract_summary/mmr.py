from chinese_sentence_cut import cut_sent
import jieba
from stopwords.stop_words import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import copy


def mmr(text, num=8, alpha=0.6,stopword_path='stopwords/cn_stopwords.txt'):

    if type(text) == str:
        sentences = cut_sent(text)
    elif type(text) == list:
        sentences = text
    else:
        raise RuntimeError("text type must be list or a")

    bow_matrix = CountVectorizer(tokenizer=jieba.lcut,stop_words=stop_words(file_name=stopword_path)).fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)

    SimMatrix = (normalized * normalized.T).A
    # 输入文本句子长度
    len_sen = len(sentences)


    # 句子标号
    sen_idx = [i for i in range(len_sen)]
    summary_set = []
    mmr = {}
    for i in range(len_sen):
        if not sentences[i] in summary_set:
            sen_idx_pop = copy.deepcopy(sen_idx)
            sen_idx_pop.pop(i)
            # 两两句子相似度
            sim_i_j = [SimMatrix[i, j] for j in sen_idx_pop]
            score_tfidf = normalized[i].toarray()[0].sum() # / sen_word_len[i], 如果除以词语个数就不准确
            mmr[sentences[i]] = alpha * score_tfidf - (1 - alpha) * max(sim_i_j)
            summary_set.append(sentences[i])


    score_sen = [(rc[1], rc[0]) for rc in sorted(mmr.items(), key=lambda d: d[1], reverse=True)]
    if len(mmr) > num:
        score_sen = [i[1] for  i in score_sen[0:num]]
    return ("").join(score_sen)


if __name__ == '__main__':
    doc = "PageRank算法简介。" \
          "是上世纪90年代末提出的一种计算网页权重的算法! " \
          "当时，互联网技术突飞猛进，各种网页网站爆炸式增长。 " \
          "业界急需一种相对比较准确的网页重要性计算方法。 " \
          "是人们能够从海量互联网世界中找出自己需要的信息。 " \
          "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。 " \
          "Google把从A页面到B页面的链接解释为A页面给B页面投票。 " \
          "Google根据投票来源甚至来源的来源，即链接到A页面的页面。 " \
          "和投票目标的等级来决定新的等级。简单的说， " \
          "一个高等级的页面可以使其他低等级页面的等级提升。 " \
          "具体说来就是，PageRank有两个基本思想，也可以说是假设。 " \
          "即数量假设：一个网页被越多的其他页面链接，就越重）。 " \
          "质量假设：一个网页越是被高质量的网页链接，就越重要。 " \
          "总的来说就是一句话，从全局角度考虑，获取重要的信。 "
    sum = mmr(doc)
    print(sum)





