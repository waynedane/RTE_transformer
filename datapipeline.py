import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

label_dict = {  "entailment": 0,
                 "neutral": 1,
                 "contradiction": 2}

import jsonlines
def load_data_from_file(path):
    with open(path,mode='r+',encoding='utf8') as f:
        data_list=[]
        for step,item in enumerate(jsonlines.Reader(f)):
            if item['gold_label']=='-':
                continue
            data_list.append({'index':step,'sentence1':item['sentence1'],'sentence2':item['sentence2'],'gold_label':item['gold_label']})
    return data_list
    
dataset = SNLI_Dataset('data/SNLI/snli_1.0_train.jsonl')
dataloader = DataLoader(dataset, batch_size=10,collate_fn = collate_fn)
a_ids, a_mask, b_ids, b_mask ,l = next(iter(dataloader))
