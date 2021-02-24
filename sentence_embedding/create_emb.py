#Create sentences embeddings
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import pandas as pd
import json
from textblob import TextBlob
import nltk
from scipy import spatial
import torch
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
#pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

train = pd.read_json("data/train-v1.1.json")
valid = pd.read_json("data/dev-v1.1.json")

contexts = []
questions = []
answers_text = []
answers_start = []
for i in range(train.shape[0]):
    topic = train.iloc[i,0]['paragraphs']
    for sub_para in topic:
        for q_a in sub_para['qas']:
            questions.append(q_a['question'])
            answers_start.append(q_a['answers'][0]['answer_start'])
            answers_text.append(q_a['answers'][0]['text'])
            contexts.append(sub_para['context'])   
df = pd.DataFrame({"context":contexts, "question": questions, "answer_start": answers_start, "text": answers_text})
df.to_csv("data/train.csv", index = None)

#Create dictionary of sentence embeddings
from models import InferSent

paras = list(df["context"].drop_duplicates().reset_index(drop= True))
blob = TextBlob(" ".join(paras))
sentences = [item.raw for item in blob.sentences]

MODEL_PATH = 'InferSent/encoder/infersent1.pkl'
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = 'InferSent/GloVe/glove.840B.300d.txt'
infersent.set_w2v_path(W2V_PATH)

infersent.build_vocab(sentences, tokenize=True)
dict_embeddings = {}
for i in range(len(sentences)):
    print(i)
    dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)
    
questions = list(df["question"])
for i in range(len(questions)):
    print(i)
    dict_embeddings[questions[i]] = infersent.encode([questions[i]], tokenize=True)
    
d1 = {key:dict_embeddings[key] for i, key in enumerate(dict_embeddings) if i % 2 == 0}
d2 = {key:dict_embeddings[key] for i, key in enumerate(dict_embeddings) if i % 2 == 1}

with open('data/dict_embeddings1.pickle', 'wb') as handle:
    pickle.dump(d1, handle)
    
with open('data/dict_embeddings2.pickle', 'wb') as handle:
    pickle.dump(d2, handle)