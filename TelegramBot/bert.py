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
from context_similarity import *
import pickle

args = pickle.load(open("context_similarity/context_vectors.sav", "rb"))

class BertHandler():

    def __init__(self):
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


    def handle(self, question):
        context = select_context(question, *args)

        def answer_question(question, answer_text, tokenizer=self.tokenizer, model=self.model):

            # Apply the tokenizer to the input text, treating them as a text-pair.
            input_ids = tokenizer.encode(question, answer_text)
            # Report how long the input sequence is.
            print('Query has {:,} tokens.\n'.format(len(input_ids)))
            # Search the input_ids for the first instance of the token.
            sep_index = input_ids.index(tokenizer.sep_token_id)
            # The number of segment A tokens includes the sep token istelf.
            num_seg_a = sep_index + 1
            # The remainder are segment B.
            num_seg_b = len(input_ids) - num_seg_a
            # Construct the list of 0s and 1s.
            segment_ids = [0]*num_seg_a + [1]*num_seg_b
            # There should be a segment_id for every input token.
            assert len(segment_ids) == len(input_ids)
            outputs = model(torch.tensor([input_ids]), 
                            token_type_ids=torch.tensor([segment_ids]), 
                            return_dict=True) 

            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            # Find the tokens with the highest `start` and `end` scores.
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)
            # Get the string versions of the input tokens.
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            # Start with the first token.
            answer = tokens[answer_start]
            # Select the remaining answer tokens and join them with whitespace.
            for i in range(answer_start + 1, answer_end + 1):
                # If it's a subword token, then recombine it with the previous token.
                if tokens[i][0:2] == '##':
                    answer += tokens[i][2:]
                # Otherwise, add a space then the token.
                else:
                    answer += ' ' + tokens[i]

            return answer

        msg = answer_question(question, context)

        print(context)

        return f"bert: {msg}" 