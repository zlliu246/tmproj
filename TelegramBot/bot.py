import logging
import torch
import textwrap
import requests
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import os
import re
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
PORT = int(os.environ.get('PORT', 5000))
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
TOKEN = '1754891196:AAGY6A1tFgRa2vkrzxQbK5rYT1jxkhr84ZM'
PARAGRAPH, QUESTION = range(2)
cancel_keyboard = [['Cancel']]
model_keyboard = [['Bert','XXX','YYY']]
model_markup = ReplyKeyboardMarkup(model_keyboard)
cancel_markup = ReplyKeyboardMarkup(cancel_keyboard)

def start(update, context):
    update.message.reply_text('Description Text')

def help(update, context):
    update.message.reply_text('Assistance Text')

def cancel(update,context):
    update.message.reply_text('Canceled command')
    return ConversationHandler.END

def bert(update,context):
    update.message.reply_text("BERT is chosen")

def bertQA(update, context):
    wrapper = textwrap.TextWrapper(width=80) 
    paragraph = update.message.text
    return paragraph
    #"Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the 'golden anniversary' with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as 'Super Bowl L'), so that the logo could prominently feature the Arabic numerals 50."
    

def question(update, context):
    question = update.message.text
    answer= answer_question(question,bertQA)
    update.message.reply_text(answer)

def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def answer_question(question, answer_text):
    input_ids = tokenizer.encode(question, answer_text)
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    outputs = model(torch.tensor([input_ids]), 
                    token_type_ids=torch.tensor([segment_ids]), 
                    return_dict=True) 
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]
    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    return answer

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start',start),CommandHandler('bert', bert), MessageHandler(Filters.text, bertQA), MessageHandler(Filters.text,question)],
        states = {
            PARAGRAPH : [MessageHandler(Filters.text,bertQA)],
            QUESTION : [MessageHandler(Filters.text,question)]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    dp.add_handler(conv_handler)
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler('cancel',cancel))
    dp.add_error_handler(error)
    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=TOKEN)
    updater.bot.setWebhook('https://twinkle-qna.herokuapp.com/' + TOKEN)
    updater.idle()

if __name__ == '__main__':
    main()