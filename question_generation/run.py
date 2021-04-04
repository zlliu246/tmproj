from pipelines import pipeline
from time import time
import pandas as pd
import json

ts=[]
def cp(text="", ts=ts):
    now = time()
    if len(ts) == 0:
        print("start")
        ts.append(now)
    else:
        print(text, round(now-ts[-1],4), "seconds")
        ts.append(now)

cp()

docs = pd.read_csv("../legal_squad_data.csv").loc[:,"context"].drop_duplicates().values
cp("loading dataset")

nlp = pipeline("question-generation");cp("loading pipeline")

with open("generated.txt", "w") as f:
    for doc in docs:
        try:
            qa = nlp(doc)
            line = json.dumps(qa) + "\t" + doc
            print(line)
            
            f.write(str(line) + "\n")
            print("SUCCESS")

        except Exception as err:
            print("ERROR -", err)

        cp("iteration")

