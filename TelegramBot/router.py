from TelegramBot.handlers import *

class Router():

    def __init__(self):
        self.map = {

        }

    def route(self, update, context):
        text = update.message.text

        handler = Handler()

        update.message.reply_text(handler.handle(text))

