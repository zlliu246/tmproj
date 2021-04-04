from time import time
times = []
def cp(text="",times=times):
    now = time()
    if len(times) == 0:
        print("start")
        times.append(now)
    else:
        print(text, "-->", round(now-times[-1],3), "seconds")
        times.append(now)
cp()

from TelegramBot.berthandler.document_retrieval import *; cp("importing berthandler.document_retrieval")
from TelegramBot.berthandler.bert import *; cp("berthandler.bert")

class BertHandler():

    def __init__(self):
        self.df = pickle.load(open("data/legal_doc_retrieval_cleaned_3_apr.pkl", "rb"))

    def handle(self, question):
        bestcontext = ensemble_doc_retrieval(self.df, question)
        return answer_question(question, bestcontext)


