import numpy as np
import pandas as pd

data = pd.read_csv("../data/legal_squad_data.csv").loc[:,["context","context_id"]]

generated = pd.read_csv("generated.txt", sep="\t", header=None)

d = {}
for c,cid in data.values:
    d[c] = cid

generated.columns = ["qapairs", "context"]

generated["context_id"] = [d.get(c,-1) for c in generated["context"]]

generated.to_csv("generated.csv", sep="\t")