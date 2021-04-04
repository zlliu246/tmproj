from TelegramBot.document_retrieval import *
from TelegramBot.berthandler import *
from TelegramBot.POSTaggingHandler import *
from TelegramBot.naivebayes import *
from TelegramBot.qn_suggestion import *

"""
IMPORTANT: run the java stanford parser before running tele bot
"""

class Router():

    def __init__(self):
        self.df = pickle.load(open("data/legal_doc_retrieval_cleaned_3_apr.pkl", "rb"))
        self.map = {
            "/pos": RuleBasedPOSTaggingHandler(),
            "/bert": BertHandler(),
            "/nb": NaiveBayesHandler()
        }
        self.questionSuggestionHandler = QuestionSuggestionHandler()
        print("router has been initialized\n")

    def route(self, update, context):
        text = update.message.text
        reply = self.process_user_input(text)
        update.message.reply_text(reply)

        # with open("LOG.txt", "a") as f:
        #     f.write(f"{cmd}\t{text}\t{reply}\n")

    def process_user_input(self, text):
        try:
            cmd, *text = text.split(" ")
            question = " ".join(text)

            if cmd not in self.map:
                question = cmd + " " + question
                cmd = "/bert"

            # do document retrieval first before feeding to models
            bestcontexts = ensemble_doc_retrieval(self.df, question)

            reply = self.map[cmd].handle(question, bestcontexts[0][-1])

            suggested_questions = self.questionSuggestionHandler.suggest(text, bestcontexts)

            reply += "\n\n" + "Suggested Questions:\n"

            for i,q in enumerate(suggested_questions):
                reply += f"{i+1}) {q}\n"
            
            return reply

        except Exception as err:
            return str(err)