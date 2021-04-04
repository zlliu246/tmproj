"""
this script generates cache.sav
    - contains vectors computed from questions
    - Rationale: so that our question similarity algo doesn't take 10 years to run (hopefully)
    - TDIDF might really take 10 years given 160k questions to recompute and recompute again
"""

from time import time
import json
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

def parse_generated_qns():
    """
    generated questions are split into a.txt and b.txt because >50mb & cannot push to git lol
    """
    def read_file(filename):
        with open(filename, encoding="iso-8859-1") as f:
            return f.read().split("\n")

    raw = read_file("generated_qns/a.txt") + read_file("generated_qns/b.txt")

    questions, answers = [], []
    for line in raw:
        try:
            qa, context = line.split("\t")
            qa = json.loads(qa)
            
            questions.append(qa["question"])
            answers.append(qa["answer"])

        except:pass

    return questions, answers

data = pd.read_csv('../data/SQuAD_csv.csv').loc[:, ['question','text']].rename(columns={"text":"answer"}, inplace=False)
data_questions = list(data.loc[:,"question"].values)
data_answers = list(data.loc[:,"answer"].values)

gen_questions, gen_answers = parse_generated_qns()

questions = data_questions + gen_questions
answers = data_answers + gen_answers

print("no. of questions:", len(questions), "no. of answers:", len(answers))

vectorizer = TfidfVectorizer(decode_error="ignore", stop_words="english")
question_vectors = vectorizer.fit_transform(questions)

import pickle 

pickle.dump((vectorizer, question_vectors, questions, answers), open("model.sav", "wb"))

