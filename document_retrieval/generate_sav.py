"""
generate sav containing vectors for each context
"""

import numpy as np
import pandas as pd
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

data = set(pd.read_csv("../data/SQuAD_csv.csv", encoding="utf-8").loc[:,"context"].values)

print(len(data), "unique contexts in total")

stop_words = nltk.corpus.stopwords.words("english")

def clean(text):
    text = ' '.join([word.lower() for word in text.split() if word not in stop_words])
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s{2,}', " ", text)
    return text

data = [clean(i) for i in data]

vectorizer = TfidfVectorizer(decode_error="ignore", stop_words="english")
X = vectorizer.fit_transform(data)

import pickle

pickle.dump((vectorizer, X, data), open("context_vectors.sav", "wb"))

