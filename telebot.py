from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, Dispatcher
from TelegramBot.router import *

TOKEN='1754891196:AAGY6A1tFgRa2vkrzxQbK5rYT1jxkhr84ZM'

updater = Updater(TOKEN, use_context=True)
dispatcher = updater.dispatcher

router = Router()
dispatcher.add_handler(MessageHandler(Filters.text, router.route))

updater.start_polling()

print()
print("telegrambot is running!")

updater.idle()
