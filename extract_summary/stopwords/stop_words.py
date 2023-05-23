
def stop_words(file_name='stopwords/cn_stopwords.txt'):
    with open(file_name,'r',encoding='utf-8') as f:
        stopwords=f.read().splitlines()

    return stopwords