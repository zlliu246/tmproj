from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, Dispatcher
from TelegramBot.router import *

"""
IMPORTANT!!!!!: Do NOT push our telebot token to git plz
also, create a .env in your root directory containing just the token
"""

with open(".env") as f:
    TOKEN = f.read().strip()

updater = Updater(TOKEN, use_context=True)
dispatcher = updater.dispatcher

router = Router()
dispatcher.add_handler(MessageHandler(Filters.text, router.route))

updater.start_polling()

print()
print("halobot is running!")

updater.idle()
