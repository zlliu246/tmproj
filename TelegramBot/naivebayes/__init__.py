from TelegramBot.naivebayes.helper import *

class NaiveBayesHandler():
    def __init__(self):
        pass

    def handle(self, question, context):
        return answer_question(context, question).strip()
        