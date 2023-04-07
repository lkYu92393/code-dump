import argparse
import os
import requests
import json

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--output', default="./bert-qa")
args = parser.parse_args()

MODEL_DIR = args.output

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


valid_answers = apply_end_index(valid_answers, valid_contexts)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


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


valid_encodings = encode_data(valid_contexts, valid_questions, valid_answers)
del valid_contexts, valid_questions, valid_answers


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict) -> None:
        self.encodings = encodings

    def __getitem__(self, index: int) -> dict:
        return {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids'])


valid_ds = SquadDataset(valid_encodings)

del valid_encodings

print("FINISH PREPING DATASET. MOVE TO EVALUATION")

tokenizer = BertForQuestionAnswering.from_pretrained(MODEL_DIR)
model = BertForQuestionAnswering.from_pretrained(MODEL_DIR)

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