import logging
import torch
import textwrap
import requests
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from transformers import pipeline
import os
import re
from context_similarity import *
import pickle


args = pickle.load(open("context_similarity/context_vectors.sav", "rb"))

class BertHandler():

    def __init__(self):
        pass

    def handle(self, question):
        context = select_context(question, *args)
        nlp = pipeline("question-answering")

        def answer_question(question,answer_text):
            result = nlp(question=question, context=answer_text)
            return result['answer']

        msg = answer_question(question, context)

        print(context)
        return f"bert: {msg}" 