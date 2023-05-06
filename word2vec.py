from gensim.test.utils import common_texts
from gensim.models import Word2Vec

import re
import numpy as np
import pandas as pd
from nltk import word_tokenize

embedding_width = 128

def clean_str(string):
    """
    Tokenization/string cleaning. 
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()  # .lower() word2vec is case sensitive

df = pd.read_csv('./MIMIC-III.csv')
df = df[df.category == 'Discharge summary']
# tokenizer = StanfordTokenizer()
# df = df.head(1000)
df['text'] = df.apply(lambda x: clean_str(x['text']).split(' '), axis= 1)
print(len(df))
print(str(df.head(1).text.values))

w2v_model = Word2Vec(df.text.values, window=10, min_count=5, negative=10, sg=0, iter=15, seed=0)
# w2v_model.similar_by_word('nausea', topn=10)
w2v_model.wv.save("mimiciii_word2vec.wordvectors")
