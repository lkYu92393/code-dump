import argparse
import os, requests


import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--model', default="bert-large-uncased-whole-word-masking-finetuned-squad")
parser.add_argument('--output', default="./bert-qa")
args = parser.parse_args()


model_name = args.model
output = args.output

DATA_DIR = "./"
url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

    # loop through
    for filename in ['train-v2.0.json', 'dev-v2.0.json']:
        # make the request to download data over HTTP
        res = requests.get(f'{url}{filename}')
        # write to file
        with open(f'{os.path.join(DATA_DIR, filename)}', 'wb') as f:
            for chunk in res.iter_content(chunk_size=4):
                f.write(chunk)

        print(f"{filename} downloaded.")

import os, json


def read_squad_json(filename: str) -> tuple:
    """
    Give the datapath (representing train or dev set of SQuAD 2.0) and return the contexts, questions and answers
    """
    path = os.path.join(DATA_DIR, filename)
    with open(path, "rb") as json_file:
        squad_dict = json.load(json_file)
    
    contexts, questions, answers = list(), list(), list()
    
    # # iterate through all data in squad data
    for sample in squad_dict['data']:
        for passage in sample['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                # check if we need to be extracting from 'answers' or 'plausible_answers'
                access = "plausible_answers" if "plausible_answers" in qa.keys() else 'answers'
                for answer in qa[access]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    
    return contexts, questions, answers


train_contexts, train_questions, train_answers = read_squad_json('train-v2.0.json')
valid_contexts, valid_questions, valid_answers = read_squad_json('dev-v2.0.json')

def apply_end_index(answers: list, contexts: list) -> list:
    '''
    the dataset has already character start_index of answers' 
    '''
    _answers = answers.copy()
    for answer, context in zip(_answers, contexts):
        # this is the answer which is extracted from context 
        answer_bound = answer['text']
        # we already know the start character position of answer from context
        start_idx = answer['answer_start']
        
        answer['answer_end'] = start_idx + len(answer_bound)
    return _answers



train_answers = apply_end_index(train_answers, train_contexts)
valid_answers = apply_end_index(valid_answers, valid_contexts)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)

def encode_data(contexts: list, questions: list, answers: list) -> dict:
    encodings = tokenizer(contexts, questions, truncation=True, padding=True, return_tensors="pt")

    # add start and end positions to encodings
    start_positions, end_positions = list(), list()

    for index in range(len(answers)):
        start_value = encodings.char_to_token(index, answers[index]['answer_start'])
        end_value   = encodings.char_to_token(index, answers[index]['answer_end'])

        # if start position is None, the answer passage has been truncated
        if start_value is None:
            start_value = tokenizer.model_max_length
        
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_value is None:
            end_value = encodings.char_to_token(index, answers[index]['answer_end'] - shift)
            shift += 1

        start_positions.append(start_value)
        end_positions.append(end_value)

    encodings.update({
        'start_positions': start_positions, 'end_positions': end_positions
    })

    return encodings



train_encodings = encode_data(train_contexts, train_questions, train_answers)
valid_encodings = encode_data(valid_contexts, valid_questions, valid_answers)

train_encodings.keys()

del train_contexts, train_questions, train_answers
del valid_contexts, valid_questions, valid_answers

import torch


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict) -> None:
        self.encodings = encodings

    def __getitem__(self, index: int) -> dict:
        return {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids'])


train_ds = SquadDataset(train_encodings)
valid_ds = SquadDataset(valid_encodings)

del train_encodings, valid_encodings


#prepare BERT model and tokenizer
if os.path.exists(output):
    model = BertForQuestionAnswering.from_pretrained(output)
    print("EXISTING MODEL FOUND")
else:
    model = BertForQuestionAnswering.from_pretrained(model_name)
    print("PULLING NEW MODEL FROM WEB")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# setup GPU/CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# move model over to detected device
model.to(device)
# activate training mode of model
model.train()


from transformers import AdamW

# initialize adam optimizer with weight decay (reduces chance of overfitting)
optim = AdamW(model.parameters(), lr=5e-5)


from torch.utils.data import DataLoader
from tqdm import tqdm

import warnings
warnings.simplefilter("ignore")


# initialize data loader for training data
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)


for epoch in range(3):
    # set model to train mode
    model.train()
    
    # setup loop (we use tqdm for the progress bar)
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        
        # pull all the tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        # train model on batch and return outputs (incl. loss)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        start_positions=start_positions, end_positions=end_positions)
        # extract loss
        loss = outputs[0]
        # calculate loss for every parameter that needs grad update
        loss.backward()
        
        # update parameters
        optim.step()
        
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)


from torch.utils.data import DataLoader


# switch model out of training mode
model.eval()
model = model.to(device)

# initialize validation set data loader
val_loader = DataLoader(valid_ds, batch_size=16)

# initialize list to store accuracies
acc = list()

# loop through batches
for batch in val_loader:
    # we don't need to calculate gradients as we're not training
    with torch.no_grad():
        # pull batched items from loader
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # we will use true positions for accuracy calc
        start_true = batch['start_positions'].to(device)
        end_true = batch['end_positions'].to(device)
        
        # make predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        # pull prediction tensors out and argmax to get predicted tokens
        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        
        # calculate accuracy for both and append to accuracy list
        acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
        acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
        
# calculate average accuracy in total
print(f"Score of the model based on EM: {sum(acc)/len(acc)}") 