from TelegramBot.bert import BertHandler
from TelegramBot.cosine_similarity import CosineSimilarityHandler

class Router():

    def __init__(self, framework="/bert"):
        self.framework = framework
        self.framework_map = {
            "/bert": BertHandler(),
            "/cossim": CosineSimilarityHandler()
        }

    def route(self, update, context):
        text = update.message.text

        if text in self.framework_map:
            self.framework = text
            msg = f"framework has been changed to {text}!"
            update.message.reply_text(msg)
            print(msg)

            return


        handler = self.framework_map[self.framework]

        update.message.reply_text(handler.handle(text))

