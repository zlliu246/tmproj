import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
wordnet_lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

import string
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from unidecode import unidecode
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import numpy as np
import re
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
warnings.filterwarnings("ignore")

import math
import re
from collections import Counter

import operator
from nltk.parse import stanford
import os
from nltk.parse.corenlp import CoreNLPParser

os.environ['STANFORD_MODELS'] = '/Users/lzl/Documents/apps/stanford-parser'
os.environ['STANFORD_PARSER'] = '/Users/lzl/Documents/apps/stanford-parser'

parser = CoreNLPParser("http://localhost:9000")
WORD = re.compile(r"\w+")

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def answer_question(query, context):
    query = re.sub(r'[^\w\s]', '', query)
    result = next(parser.raw_parse((query).lower()))
    query_split = query.split(" ")
    sub_clause_labels = ['NP', 'N', 'VP', 'PP']
    new_query = ''

    if query_split[0].lower() in ['what', 'when', 'why', 'how', 'where', 'which', 'who']:
            if subtree.label() in sub_clause_labels:
                new_query += ' '.join(subtree.leaves()) + " "
    else:
        for subtree in result:
            if subtree.label() in sub_clause_labels:
                new_query += ' '.join(subtree.leaves()) + " "

    if len(new_query) == 0:
        new_query += query.lower()
    
    context = context.lower()
    context_split = re.split('[!#$%&\()*+-./:;<=>?@[\\]^_`{|}~]', context)
    for i in range(len(context_split)):
        context_split[i] = context_split[i].strip()

    context_split = [x for x in context_split if len(x)>0]
    
    clauses = []
    labels = ['SBAQ', 'S', 'SQ', 'SBAR', 'FRAG', 'SBARQ']
    for cs in context_split:
        result = next(parser.raw_parse(cs))
        if result[0].label() in labels:
            clauses.append(' '.join(result[0].leaves()))

    clauses_dictionary = {}
    clause_counter = 0
    np_dictionary = {}

    for clause in clauses:
        result = next(parser.raw_parse(clause))
        if result[0].label() in labels:
            clause_counter += 1
            clauses_dictionary[str(clause_counter)] = {}
            clauses_dictionary[str(clause_counter)][str(clause_counter) + "_" + "clause"] = ' '.join(result[0].leaves())
            np_dictionary[str(clause_counter) + "_NVP_" + "clause"] = ''
            inner_clause1 = 0

            for subtree in result[0]:
                if subtree.label() in labels:
                    inner_clause1 += 1
                    clauses_dictionary[str(clause_counter)][str(clause_counter) + "_" + str(inner_clause1) + "_" + "clause"] = ' '.join(subtree.leaves())
                    np_dictionary[str(clause_counter) + "_" + str(inner_clause1) + "_NVP_" + "clause"] = ''

                    inner_clause2 = 0
                    for subtree2 in subtree:
                        if subtree2.label() in labels:
                            inner_clause2 += 1
                            clauses_dictionary[str(clause_counter)][str(clause_counter) + "_" + str(inner_clause1) + "_" + str(inner_clause2) + "_" + "clause"] = ' '.join(subtree2.leaves())
                            np_dictionary[str(clause_counter) + "_" + str(inner_clause1) + "_" + str(inner_clause2) + "_NVP_" + "clause"] = ""

                            for subtree3 in subtree2:
                                np_dictionary[str(clause_counter) + "_" + str(inner_clause1) + "_" + str(inner_clause2) + "_NVP_" + "clause"] += ' '.join(subtree3.leaves()) + " "

                        elif subtree2.label() in sub_clause_labels:
                            np_dictionary[str(clause_counter) + "_" + str(inner_clause1) + "_NVP_" + "clause"] += ' '.join(subtree2.leaves()) + " "

                elif subtree.label() in sub_clause_labels:
                    np_dictionary[str(clause_counter) + "_NVP_" + "clause"] += ' '.join(subtree.leaves()) + " "

    question_np = new_query
    question_vector = text_to_vector(question_np)
    cosine_np_dict = {}

    for np in np_dictionary:
        np_vector = text_to_vector(np_dictionary[np])
        cosine_np_dict[np] = get_cosine(question_vector, np_vector)
        
    max_clause_id, max_sim = max(cosine_np_dict.items(), key=operator.itemgetter(1))[0], max(cosine_np_dict.items(), key=operator.itemgetter(1))[1]

    clause_id = max_clause_id[:max_clause_id.find("NVP")-1]

    if max_sim >= 0.2:
        clause_id = max_clause_id[:max_clause_id.find("NVP")-1]
        first_clause_id = clause_id[:clause_id.find("_")]

        answer = clauses_dictionary[first_clause_id][clause_id + "_clause"]
        answer_split = answer.split(" ")
        if answer_split[0] in ["and", 'but', 'or', 'nor', 'after', 'before', 'although']:
            answer_split.pop(0)
        answer_return = " ".join(answer_split)

    return answer_return

        