from TelegramBot.document_retrieval import *
from TelegramBot.berthandler import *
from TelegramBot.POSTaggingHandler import *

class Router():

    def __init__(self):
        self.df = pickle.load(open("data/legal_doc_retrieval_cleaned_3_apr.pkl", "rb"))
        self.map = {
            "/pos": RuleBasedPOSTaggingHandler(),
            "/bert": BertHandler()
        }
        # self.questionSuggestionHandler = QuestionSuggestionHandler(df)

    def route(self, update, context):
        text = update.message.text
        cmd, *text = text.split(" ")
        question = " ".join(text)

        if cmd not in self.map:
            cmd = "/pos"

        # do document retrieval first before feeding to models
        bestcontexts = ensemble_doc_retrieval(self.df, question)

        reply = self.map[cmd].handle(question, bestcontexts[0][-1])

        # suggested_questions = self.questionSuggestionHandler.suggest(text)

        update.message.reply_text(reply)

        # with open("LOG.txt", "a") as f:
        #     f.write(f"{cmd}\t{text}\t{reply}\n")
