"""
Run this file to test telegram bot functionality without
having to run the telegram bot
"""

from TelegramBot.router import *

router = Router()

while True:
    userinput = input("Enter your query here >>> ")
    print(router.process_user_input(userinput))

    print("="*100)
