import json
import pandas as pd

data = pd.read_csv("out.txt", delimiter="\t", header=None).iloc[:,0].values

n = 0
for qaj in data:
    qa = json.loads(qaj)

    try:
        assert type(qa) == list
        assert type(qa[0]) == dict
        assert "answer" in qa[0]
        assert "question" in qa[0]

        n += len(qa)

    except:pass

print(n)
