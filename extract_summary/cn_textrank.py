import networkx as nx
import jieba
from extract_summary.chinese_sentence_cut import cut_sent
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
 
def textRank(document,rate=0.3,nums=None):

    sentences = cut_sent(document)
 
    bow_matrix = CountVectorizer(tokenizer=jieba.lcut).fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
 
    similarity_graph = normalized * normalized.T

    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)

    text_rank_graph = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    number_of_nodes = int(rate*len(text_rank_graph))

    if nums==None:
        if number_of_nodes < 3:
            number_of_nodes = 3
    else:
        if type(nums)==int:
            number_of_nodes = nums
    del text_rank_graph[number_of_nodes:]
    
    summary = ' '.join(word for _,word in text_rank_graph)
    
    return summary

if __name__=='__main__':
    text_example="""基于bert的中文自然语言处理工具。包括情感分析、中文分词、词性标注、以及命名实体识别功能。提供了训练接口，通过指定输入输出以及谷歌提供的下载好的预训练模型即可进行自己的模型的训练，训练任务有task_name参数决定，目前提供的任务主要包括句子匹配、文本分类、命名实体识别、序列标注任务。使用pip install tudou安装使用。需要下载预先训练好的模型，模型地址在底部"""

    print(textRank(text_example))