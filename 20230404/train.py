import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm  # for showing progress bar
from datasets import load_dataset


import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--model', default="bert-large-uncased-whole-word-masking-finetuned-squad")
parser.add_argument('--output', default="./bert-qa")
args = parser.parse_args()


device = torch.device('cuda:0')
#Using torch by GPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

task = "./data/formatted-SQuAD-train.csv"

#Download the dataset from SQuAD
#SQuAD = pd.read_json('./data/train-v2.0.json')
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
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
        new_answers.append(answer)
    return new_answers

def prep_data(dataset):
    questions = dataset['question']
    contexts = dataset['context']
    answers = add_end_idx(
        dataset['answers'],
        contexts
    )
    return {
        'question': questions,
        'context': contexts,
        'answers': answers
    }


#splict the set in train and validate
dataset = prep_data(SQuAD['train'])
vailset = prep_data(SQuAD['validation'])
#dataset_validation = prep_data(SQuAD['validation'])
print('{:>5,} training samples'.format(len(dataset['question'])))
print('{:>5,} validation samples'.format(len(vailset['question'])))


model_name = args.model
output = args.output
# model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

#prepare BERT model and tokenizer
if os.path.exists(output):
    model = BertForQuestionAnswering.from_pretrained(output)
    print("EXISTING MODEL FOUND")
else:
    model = BertForQuestionAnswering.from_pretrained(model_name)
    print("PULLING NEW MODEL FROM WEB")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

#print(dataset['answers'][:5])


# tokenize
train = tokenizer(dataset['context'],
                  dataset['question'],
                  add_special_tokens=True,
                  truncation=True,
                  return_attention_mask=True,  # Construct attn. masks.
                  padding='max_length',
                  max_length=512, return_tensors='pt')

#print(tokenizer.decode(train['input_ids'][0])[:855])

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


#training
import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# build datasets for both our training data
train_dataset = SquadDataset(train)

batch_size = 4

loader = torch.utils.data.DataLoader(train_dataset,
                                     batch_size=4,
                                     shuffle=True)

from transformers import AdamW

model.to(device)
model.train()
optim = AdamW(model.parameters(), lr=5e-5)

loop = tqdm(loader)
for batch in loop:
    optim.zero_grad()

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions)

    loss = outputs[0]
    loss.backward()
    optim.step()

    loop.set_postfix(loss=loss.item())

model.save_pretrained(output)
