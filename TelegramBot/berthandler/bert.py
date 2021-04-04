from time import time
berttimes = []
def cp(text="",times=berttimes):
    now = time()
    times = berttimes
    if len(times) == 0:
        times.append(now)
    else:
        print(text, now-times[-1])
cp()

import torch
import textwrap
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# model_config = BertConfig.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', output_hidden_states=True)
# self.bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)

cp("importing stuff for bert")

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