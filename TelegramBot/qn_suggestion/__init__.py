import numpy as np
import pandas as pd
import json
import random

class QuestionSuggestionHandler():

    def __init__(self):
        df = pd.read_csv("data/legal_squad_data.csv")
        generated = pd.read_csv("question_generation/generated.csv", delimiter="\t")
        
        d = {}
        for cid,qn in df.loc[:,["context_id", "question"]].values:
            if cid not in d:
                d[cid] = []
            d[cid].append(qn)

        for cid, qajson in generated.loc[:,["context_id", "qapairs"]].values:
            if cid not in d:
                # print("?????", cid)
                pass
            else:
                qapairs = json.loads(qajson)
                d[cid].extend([qa["question"] for qa in qapairs])

        # self.dict key=context_id, value=list of questions for context            
        self.dict = d

    def suggest(self, question, bestcontexts):
        qnpool = []
        for i,c in bestcontexts[:3]:
            qnpool.extend(self.dict[i])

        return random.sample(qnpool, 5)
