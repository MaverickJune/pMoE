import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parameter import Parameter
from collections import defaultdict

import os

from torch.distributed.device_mesh import init_device_mesh

import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, TransfoXLLMHeadModel
from typing import Dict

import logging
import urllib3
from transformers import logging as hf_logging
from datasets import logging as ds_logging

# Set global logging level
logging.basicConfig(level=logging.WARNING)

# Suppress specific libraries
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

# Suppress urllib3 warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Suppress Hugging Face library logs
hf_logging.set_verbosity_warning()
ds_logging.set_verbosity_warning()

class pMOEdataset(Dataset):
    def __init__(self, dataset_name, model_name):
        MODEL_LIST = ["transfo-xl/transfo-xl-wt103", "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-3.1-70B-Instruct", "eastwind/tinymix-8x1b-chat", "deepseek-ai/deepseek-moe-16b-base"]
        assert model_name in MODEL_LIST, f"Model name should be one of {MODEL_LIST}"
        
        DATASET_LIST = ["wikitext-2", "wikitext-103", "enwik8", "squad"]
        assert dataset_name in DATASET_LIST, f"Dataset name should be one of {DATASET_LIST}"
        
        if dataset_name == "wikitext-2":
            dataset = load_dataset("Salesforce/wikitext","wikitext-2-v1",split="train")
        elif dataset_name == "wikitext-103":
            dataset = load_dataset("Salesforce/wikitext","wikitext-103-v1",split="train[:10%]")
        elif dataset_name == "enwik8":
            dataset = load_dataset("LTCB/enwik8", split="train[:10%]", trust_remote_code=True)
        elif dataset_name == "squad":
            dataset = load_dataset("rajpurkar/squad", split="train[:10%]")
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is currently not supported")
        
        self.model_name = model_name
        self.hf_dataset = dataset
        
        self.text_column = ""
        if dataset_name in ["wikitext-2", "wikitext-103", "enwik8"]:
            self.text_column = "text"
        elif dataset_name == "squad":
            self.text_column = "question"
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is currently not supported")
                                      
        self.pd_dataset = self.convert_to_pd_dataset(dataset)
        
        os.environ["TRUST_REMOTE_CODE"] = "True"
        
        if model_name == "transfo-xl/transfo-xl-wt103":
            hf_name = "transfo-xl/transfo-xl-wt103"
            revision = '40a186da79458c9f9de846edfaea79c412137f97'
            
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name, revision=revision)
            # model = TransfoXLModel.from_pretrained(hf_name, revision=revision)
            # self.emb = model.word_emb
            if self.tokenizer.pad_token is None:
                pad = "<|pad|>"
                self.tokenizer.add_special_tokens({"pad_token": pad})
                self.tokenizer.padding_side = "left"
        elif(self.model_name=="mistralai/Mistral-7B-v0.1"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                pad = "<|pad|>"
                self.tokenizer.add_special_tokens({"pad_token": pad})
                self.tokenizer.padding_side = "left"
        elif(self.model_name=="meta-llama/Llama-3.1-70B-Instruct"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                pad = "<|pad|>"
                self.tokenizer.add_special_tokens({"pad_token": pad})
                self.tokenizer.padding_side = "left"
        elif(self.model_name=="eastwind/tinymix-8x1b-chat"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                pad = "<|pad|>"
                self.tokenizer.add_special_tokens({"pad_token": pad})
                self.tokenizer.padding_side = "left"
                
        elif(self.model_name=="deepseek-ai/deepseek-moe-16b-base"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                pad = "<|pad|>"
                self.tokenizer.add_special_tokens({"pad_token": pad})
                self.tokenizer.padding_side = "left"
        else:
            raise NotImplementedError(f"Model {model_name} is currently not supported")
        
    def convert_to_pd_dataset(self, dataset):
        row_dataset = []
        keys = list(dataset.features)
        for item in dataset:
            item_dict = {}
            for key in keys:
                item_dict[key] = item[key]
            row_dataset.append(item_dict)
            
        df = pd.DataFrame(row_dataset)
        df = df[df[self.text_column] != '']
    
        return df
    
    def get_token_lengths(self):
        def count_tokens(row: Dict, tokenizer):
            return len(tokenizer(row[self.text_column], add_special_tokens=True, return_attention_mask=True)['input_ids'])
        self.pd_dataset["num_tokens"] = self.pd_dataset.apply(lambda row: count_tokens(row, self.tokenizer), axis=1)
        
    def prune_dataset(self, max_tokens):
        self.get_token_lengths()
        self.pd_dataset = self.pd_dataset[self.pd_dataset['num_tokens'] <= max_tokens]
    
    def get_token_logistic(self):
        print(self.pd_dataset['num_tokens'].describe())
        
    def __len__(self):
        return len(self.pd_dataset)
    
    def __getitem__(self, idx):
        return self.pd_dataset.iloc[idx][self.text_column]
    
    def __getitem__(self, idx):
        text = self.pd_dataset.iloc[idx][self.text_column]
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        
        if self.model_name == "eastwind/tinymix-8x1b-chat":
            def make_prompt(instruction):
                return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            text = make_prompt(text)
            
        return text
    
def collate_fn_batching(batch, tokenizer):
    # Tokenize each item in the batch without padding
    tokenized_batch = [tokenizer(text, return_tensors="pt", add_special_tokens=True, return_attention_mask=True) for text in batch]

    # Find the maximum sequence length in the batch
    max_length = max(item['input_ids'].shape[1] for item in tokenized_batch)

    # Pad each sequence to the maximum length in the batch
    input_ids = [torch.nn.functional.pad(item['input_ids'], (max_length - item['input_ids'].shape[1], 0), value=tokenizer.pad_token_id) for item in tokenized_batch] # Pad on the left
    attention_mask = [torch.nn.functional.pad(item['attention_mask'], (max_length - item['attention_mask'].shape[1], 0), value=0) for item in tokenized_batch] # Pad on the left

    # Stack tensors to create batched tensors
    return {
        'input_ids': torch.cat(input_ids),
        'attention_mask': torch.cat(attention_mask)
    }
    
class ContextManager:
    """
    ContextManager handles the initialization of distributed processes, 
    including rank, world size, and mesh topology.

    Args:
        rank (int): Current global rank.
        world_size (int): Global world size.
        mesh_shape (tuple): Mesh dimensions for distributed communication.
        mesh_dim_names (tuple of str): Names for each mesh dimension (e.g., "data_parallel", "tensor_parallel").
        backend (str, optional): Backend for distributed communication. Default is 'nccl'.
    """

    def __init__(self, rank, world_size, mesh_shape, mesh_dim_names, backend='nccl'):
        """
        Initializes the ContextManager with the provided configuration.
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.shape_kv = dict(zip(mesh_dim_names, mesh_shape))
        
        # Initialize the device mesh
        self.mesh = init_device_mesh(
            backend,  # Hardcoded backend for device mesh initialization
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names
        )
    
    def get_pg_count(self):
        return dist.get_pg_count()

    def get_group(self,name):
        return self.mesh.get_group(name)
    
    def get_rank(self, name):
        return self.mesh.get_local_rank(name)
    
    def get_size(self, name):
        return self.shape_kv[name]


def generate_dummy_tokens(batch_size, seq_len):
    return torch.rand((batch_size, seq_len))







