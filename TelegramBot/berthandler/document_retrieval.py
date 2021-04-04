from time import time
times = []
def cp(text="",times=times):
    return
    now = time()
    if len(times) == 0:
        times.append(now)
    else:
        print(text, now-times[-1])
cp()

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
wordnet_lemmatizer = WordNetLemmatizer()
import string
import pickle

import sys
# !pip install unidecode
from unidecode import unidecode
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

warnings.filterwarnings("ignore")

from numpy.linalg import svd, norm
from nltk.stem.snowball import EnglishStemmer
from collections import defaultdict, Counter
import random

stop_words = stopwords.words("english")

cp("importing shit")

def clean_normalcase_stop_lem(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s{2,}', " ", text)
    text = unidecode(text)
    text = ' '.join([wordnet_lemmatizer.lemmatize(i) for i in text.split()])
    return text

def clean_normalcase_nostop_lem(text):
    text = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s{2,}', " ", text)
    text = unidecode(text)
    return text

def clean_lowercase_stop_lem(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s{2,}', " ", text)
    text = unidecode(text)
    text = ' '.join([wordnet_lemmatizer.lemmatize(i) for i in text.split()])
    return text.lower()

def clean_lowercase_nostop_lem(text):
    text = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
#     text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s{2,}', " ", text)
    text = unidecode(text)
    return text.lower()

def clean_lowercase_nostop(text):
    text = ' '.join([word for word in text.split() if word not in stop_words])
#     text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s{2,}', " ", text)
    text = unidecode(text)
    return text.lower()

# Query preprocessing for paragraph document retrieval
def clean_query_lem(query):
    result = re.sub('[%s]' % re.escape(string.punctuation), '', query)
    result = re.sub('\s{2,}', " ", result).lower()
    result = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in result.split() if word not in stop_words])

    if result == '':
        return None
    return result

# Query preprocessing for paragraph document retrieval
def clean_query_nolem(query):
    result = re.sub('[%s]' % re.escape(string.punctuation), '', query)
    result = re.sub('\s{2,}', " ", result).lower()
    result = ' '.join([word for word in result.split() if word not in stop_words])

    if result == '':
        return None
    return result

class LSI:
    def __repr__(self):
        return 'LSI( terms:{}, documents:{}, index_ready:{})'.format(self.index.__len__(),
                                                                     self.documents.__len__(),
                                                                     not(self.__update_index))
    
    def __init__(self, tokenizer=nltk.word_tokenize,
                 stemmer=EnglishStemmer,
                 stopwords=nltk.corpus.stopwords.words('english'),
                 variance=0.9):
        '''
        >>> queries cannot work unless __update_index is false.        
        '''
        self.stemmer = stemmer()
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        
        self.documents = {}
        self.index = defaultdict(Counter)
        
        self.A = None # term document matrix
        self.U = None # output of svd
        self.S = None # output of svd
        self.V = None # output of svd
        
        self.term_rep = np.array(None) # reduced representation of terms after svd
        self.doc_rep = np.array(None)  # reduced representation of documents after svd
        
        self.__update_index = True
        self._term_index_in_A = {}
        self._document_index_in_A = {}
        
        self.k = None # reduced dimension after SVD
        self.variance = variance # variance to retain after SVD
            
        
    def add_doc(self, document, document_id):
        '''
        add terms into vocabulary.
        add document 
        '''
        if document_id in self.documents:
            # print('document_id : {} already indexed.'.format(document_id))
            return False
        
        for token in [t.lower() for t in self.tokenizer(document) if t.isalpha()]:
            if token in self.stopwords:
                continue;
            if self.stemmer:
                token = self.stemmer.stem(token)
                
            # add this token to defaultdict(Counter)
            # this document's count is increased by 1 for this token's Counter
            self.index[token].update({document_id:1})
        
        self.__update_index = True # update flag to rebuild index
        self.documents[document_id] = document # add document to documents
        return True
    
    
    def _svd_A(self):
        '''
        Perform SVD on A and update the U,S,V matrices
        '''
        self.U, self.S, self.V = svd(self.A)
        
    
    def _get_k_for_svd(self):
        '''
        Finds the value for k after SVD such that specified variance is retained
        returns k : int
        '''
        if (self.S is not None):
            sum = 0
            k = 0
            while(sum < self.variance):
                k -=- 1
                sum = self.S[:k].sum() / self.S.sum()
            self.k = k
            return True
        else:
            # print('S is not populated.')
            return False

    def rebuild_index(self):
        '''
        >>> set _update_index to false when index is built
        '''
        terms = list(self.index.keys())
        documents = list(self.documents.keys())
        self.A = np.zeros((terms.__len__(), documents.__len__()), dtype='int8')

        self._document_index_in_A = {doc:ix for ix,doc in enumerate(documents)}
        self._term_index_in_A = {term:ix for ix,term in enumerate(terms)}
        
        for term in terms:
            counter = self.index[term]
            term_ix = self._term_index_in_A[term]
            doc_ids = list(self.index[term].keys())
            doc_vals = [counter[x] for x in doc_ids]
            doc_ixs = [self._document_index_in_A[x] for x in doc_ids]
            for ix,doc_id in enumerate(doc_ixs):
                self.A[term_ix][doc_id] = doc_vals[ix]
        print('Term-Document frequency matrix is ready.')
        print('Proceeding to do SVD on the matrix.')
        
        self._svd_A()
        self._get_k_for_svd()
        
        self.doc_rep = self.V[:self.k,:]
        self.term_rep = self.U[:,:self.k]

        print('Index Rebuilt. Setting __update_index to False. Queries can now be performed.')
        self.__update_index = False
        
    def _calc_query_doc_affinity_score(self, query_vector):
        '''
        calculates the query - document affinity score
        '''
        try:
            one_by_query_vector_norm_ = (1/norm(query_vector))
        except ZeroDivisionError:
            one_by_query_vector_norm_ = (1/1e-4)
        affinity_scores = (np.dot(query_vector,self.doc_rep) / norm(self.doc_rep, axis=0)) * one_by_query_vector_norm_
        return affinity_scores
    
    def query(self, query_string, top=5):
        query_string = clean_query_lem(query_string)
        
        if(self.__update_index == True):
            print('Index is not updated. Use rebuild_index()')
            return False
        
        query_vector = []
        for token in [t.lower() for t in self.tokenizer(query_string) if t.isalpha()]:
            if token in self.stopwords:
                continue;
            if self.stemmer:
                token = self.stemmer.stem(token)
            try:
                query_vector.append(self.term_rep[self._term_index_in_A[token], :])
            except KeyError:
                query_vector.append(np.array([0.0] * self.k))
        
        query_vector_mean = np.array(query_vector).mean(axis=0)
        affinity_scores = self._calc_query_doc_affinity_score(query_vector_mean)
              
        res_doc_index = (-affinity_scores).argsort()[:top]
        results = []
        for index in res_doc_index:
            if 'numpy.ndarray' in str(type(index)): # tie
                index = random.choice(index)
            res_doc_id = list(self._document_index_in_A.keys())[index]
            results.append(df.iloc[res_doc_id]['context_id'])
            
        return results

def build_lsi(col):
    if col == 'para':
        col = 'cleaned_lowercase_nostop_lem'
    elif col == 'summary':
        col = 'extractive_summarized_3_sent'
    else:
        print("Invalid col type. Specify 'para' or 'summary'.")
        return
    
    lsi = LSI()

    for index, row in df.iterrows():
        lsi.add_doc(row[col], index)
        
    lsi.rebuild_index()
    
    return lsi
    
def lsi(query, lsi_model):
    docs = lsi_model.query(query)
    return docs



def tfidf(df, vectorizer, tfidf_matrix, query, ner=False):
    if ner:
        query = query_to_ner_str(query)
    else:
        query = clean_query_lem(query)

    if query is None:
        return []
    
    doc_ids = get_similar_docs(df, vectorizer, tfidf_matrix, query)
    
    return doc_ids

def get_similar_docs(df, tfidfvectorizer, docs_tfidf_matrix, query):
    """
    vectorizer: TfIdfVectorizer model
    docs_tfidf: tfidf vectors for all docs
    query: query

    return: doc with highest tf-idf cosine similarity
    """
    query_tfidf = tfidfvectorizer.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf_matrix).flatten()
    max_sim = max(cosineSimilarities)
    
    if max_sim < 0.05: # not sure whether to set this threshold as some correct answers are like 0.1 similarity
#         print("No Matches")
        return []
    else:
        threshold = 0.6 * max_sim
    
    top_doc_ids = set()
    for idx, val in enumerate(cosineSimilarities):
        if val >= threshold:
            top_doc_ids.add((df.iloc[idx]['context_id'],val))
            
    top_doc_ids = sorted(top_doc_ids, key=lambda x: x[1], reverse=True)[:5]
            
#     print(f"Top Docs: {top_doc_ids}\n")
            
    return [i[0] for i in top_doc_ids]

cp()
df = pickle.load(open("data/legal_doc_retrieval_cleaned_3_apr.pkl", "rb")); cp("loading df")

# TF-IDF Vectorizer for cleaned_lowercase_nostop_lem column
para_vectorizer = TfidfVectorizer(ngram_range=(1,2)); cp("loading para vectorizer")
para_tfidf_matrix = para_vectorizer.fit_transform(df['cleaned_lowercase_nostop_lem']); cp("fit trasnform thingy")
para_lsi_model = pickle.load(open("data/para_lsi_model.sav", "rb")); cp("loading pickled para_lsi_model")

# IMPORTANT - this is gitignored cus too large to commit to git (500mb)
# para_lsi_model = build_lsi("para")
# pickle.dump(para_lsi_model, open("data/para_lsi_mode.sav", "wb"))

def ensemble_doc_retrieval(df, query):
    ranking_dict = {}
    weights = {1: 1, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6}
    
    # Para TFIDF
    tfidf_retrieved_doc_ids = tfidf(df, para_vectorizer, para_tfidf_matrix, query)
    for idx, doc_id in enumerate(tfidf_retrieved_doc_ids, start=1):
        if doc_id not in ranking_dict:
            ranking_dict[doc_id] = weights[idx]
        else:
            ranking_dict[doc_id] += weights[idx]
    
    # Para LSI
    lsi_retrieved_doc_ids = lsi(query, para_lsi_model)
    for idx, doc_id in enumerate(lsi_retrieved_doc_ids, start=1):
        if doc_id not in ranking_dict:
            ranking_dict[doc_id] = weights[idx]
        else:
            ranking_dict[doc_id] += weights[idx]
    
    results = []
    
    for doc_id, score in sorted(ranking_dict.items(), key=lambda item: item[1], reverse=True):
        res = df[df['context_id'] == doc_id]['context'].values[0]
        results.append(res)

    return results[0]

