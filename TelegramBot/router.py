from TelegramBot.berthandler import *
from TelegramBot.rulebasedpostagginghandler import *

class Router():

    def __init__(self):
        self.map = {
            "/pos": RuleBasedPOSTaggingHandler(),
            "/bert": BertHandler()
        }

    def route(self, update, context):
        text = update.message.text
        cmd, *text = text.split(" ")
        text = " ".join(text)

        if cmd not in self.map:
            cmd = "/pos"

        reply = self.map[cmd].handle(text)
        update.message.reply_text(reply)

        with open("LOG.txt", "a") as f:
            f.write(f"{cmd}\t{text}\t{reply}\n")
