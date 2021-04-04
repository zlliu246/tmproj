import numpy as np
import pandas as pd
import json
import random

class QuestionSuggestionHandler():

    def __init__(self):
        self.df = pd.read_csv("data/legal_squad_data.csv")
        self.generated = pd.read_csv("question_generation/generated.csv", delimiter="\t")

        print(self.generated)


    def suggest(self, question, bestcontexts):
        ids = [i[0] for i in bestcontexts[:3]]
        questions = self.data[[cid in ids for cid in self.df["context_id"]]].loc[:,"question"]

        return np.random.choice(questions, 3)
