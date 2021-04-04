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

from TelegramBot.berthandler.bert import *; cp("berthandler.bert")

class BertHandler():

    def __init__(self):
        pass

    def handle(self, question, bestdocument):
        return answer_question(question, bestdocument)


