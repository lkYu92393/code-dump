import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm  # for showing progress bar
from datasets import load_dataset
import json

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast

device = torch.device('cuda:0')
# Using torch by GPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

task = "./data/formatted-SQuAD-train.csv"

# Download the dataset from SQuAD
# SQuAD = pd.read_json('./data/train-v2.0.json')
SQuAD = load_dataset('squad')


def add_end_idx(answers, contexts):
    new_answers = []
    # loop through each answer-context pair
    for answer, context in tqdm(zip(answers, contexts)):
        # quick reformating to remove lists
        answer['text'] = answer['text'][0]
        answer['answer_start'] = answer['answer_start'][0]
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer['text']
        # we already know the start index
        start_idx = answer['answer_start']
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            answer['answer_end'] = end_idx
        else:
            # this means the answer is off by 1-2 tokens
            for n in [1, 2]:
                if context[start_idx - n:end_idx - n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
        new_answers.append(answer)
    return new_answers


def prep_data(dataset):
    questions = dataset['question']
    contexts = dataset['context']
    id = dataset['id']
    answers = add_end_idx(
        dataset['answers'],
        contexts
    )
    return {
        'question': questions,
        'context': contexts,
        'id': id,
        'answers': answers
    }


# splict the set in train and validate
dataset = prep_data(SQuAD['validation'])
# vailset = prep_data(SQuAD['validation'])
# dataset_validation = prep_data(SQuAD['validation'])
print('{:>5,} validation samples'.format(len(dataset['question'])))

# load trained bert
# model = torch.load('before.pt', map_location=torch.device('cpu'))

# model = torch.load('./bert_qa_pt_3', map_location=torch.device('cpu'))
model = BertForQuestionAnswering.from_pretrained('./bert_qa_pt_3/', local_files_only=True)

# BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# print(dataset['answers'][:5])


# tokenize
train = tokenizer(dataset['context'],
                  dataset['question'],
                  add_special_tokens=True,
                  truncation=True,
                  max_length=512,
                  return_attention_mask=True,  # Construct attn. masks.
                  padding='max_length',
                  return_tensors='pt')


# print(tokenizer.decode(train['input_ids'][0])[:855])

def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in tqdm(range(len(answers))):

        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


add_token_positions(train, dataset['answers'])

all_output = []
# todo loop the question

for q in range(0, len(dataset['question'])):

    print("loop : " + str(q))
    random_num = q

    question = dataset["question"][random_num]
    context = dataset["context"][random_num]
    question_answer = dataset["answers"][random_num]['text']
    question_id = dataset["id"][random_num]

    try:

        # token embedding
        input_ids = tokenizer.encode(question, context)

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # first occurence of [SEP] token
        sep_idx = input_ids.index(tokenizer.sep_token_id)
        # number of tokens in segment A - question
        num_seg_a = sep_idx + 1
        # number of tokens in segment B - text
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0] * num_seg_a + [1] * num_seg_b

        answer = ""
        # token input_ids to represent the input
        # token segment_ids to differentiate our segments - text and question
        assert len(segment_ids) == len(input_ids)

        output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)
        # print(answer_start, answer_end)

        # join the break word
        if answer_end >= answer_start:
            answer = tokens[answer_start]
            for i in range(answer_start + 1, answer_end + 1):
                if tokens[i][0:2] == "##":
                    answer = ""
                else:
                    answer += " " + tokens[i]

        if answer.startswith("[CLS]"):
            answer = "Unable to find the answer to your question."
            answer = ""

        print("Question:\n{}".format(context.capitalize()))
        print("\nQuestion:\n{}".format(question.capitalize()))
        print("\nQuestion id:\n{}".format(question_id))
        print("\nReal Answer:\n{}".format(question_answer))
        print("\nPredicted Answer:\n{}.".format(answer.capitalize()))

        temp_output = {question_id: answer}

        all_output.append(temp_output)
    except:
        temp_output = {question_id: ""}
        all_output.append(temp_output)
        print("Error")

    #if q > 100:
        #break;


with open("output.json", "w") as outfile:
    json.dump(all_output, outfile)