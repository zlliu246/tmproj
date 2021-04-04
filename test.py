
# import pandas as pd

# import pickle
# from TelegramBot.document_retrieval import *
# from TelegramBot.qn_suggestion import *

# df = pickle.load(open("data/legal_doc_retrieval_cleaned_3_apr.pkl", "rb"))

# qs = QuestionSuggestionHandler()

# qn = "I got cheated so what law should I read?"
# bestcontexts = ensemble_doc_retrieval(df, qn)

# suggestions = qs.suggest(qn, bestcontexts)

# print(suggestions)

# ========================================================================================================================

import pickle
from TelegramBot.document_retrieval import *
from TelegramBot.naivebayes import *

df = pickle.load(open("data/legal_doc_retrieval_cleaned_3_apr.pkl", "rb"))

qn = "What is tort law?"
context = ensemble_doc_retrieval(df, qn)[0][-1]

x = answer_question(context, qn)

print(x)
