from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples=load_dataset("json",data_files=data_path,split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        sample = self.samples[index]

        input_ids=self.tokenizer(str(sample["text"]),add_special_tokens=False,truncation=True,max_length=self.max_length-2)["input_ids"]
        input_ids=[self.tokenizer.bos_token_id]+input_ids+[self.tokenizer.eos_token_id]
        input_ids=input_ids+[self.tokenizer.pad_token_id]*(self.max_length-len(input_ids))
        input_ids=torch.tensor(input_ids,dtype=torch.long)

        labels=input_ids.clone()
        labels[input_ids==self.tokenizer.pad_token_id]=-100

        # attention_mask=(input_ids!=self.tokenizer.pad_token_id).long()

        return input_ids,labels

