from TelegramBot.berthandler.document_retrieval import *

from TelegramBot.berthandler.bert import *

class BertHandler():

    def __init__(self):
        self.df = pickle.load(open("data/legal_doc_retrieval_cleaned_3_apr.pkl", "rb"))

    def handle(self, question):
        bestcontext = ensemble_doc_retrieval(self.df, question)
        return answer_question(question, bestcontext)


