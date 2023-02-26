import os
import random
import numpy as np
from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
)

task_to_keys = {
    # glue
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    # "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),

    #dataset name
    "snli": ("premise", "hypothesis"),

    # local
    "trec": ("sentence", None),
    "mpqa": ("sentence", None),
    "cr": ("sentence", None),
    "sst5": ("sentence", None),
    "mr": ("sentence", None),
    "subj": ("sentence", None),
}

task_num_labels = {
    # glue
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    # "stsb": ("sentence1", "sentence2"),
    "wnli": 2,

    #dataset name
    "snli": 3,

    # local
    "trec": 6,
    "mpqa": 2,
    "cr": 2,
    "sst5": 5,
    "mr": 2,
    "subj": 2,
}


ROOT = '/home/zhuoyan/text-clf/materials'

class FSDataset(Dataset):
    def __init__(self, task_name = "mrpc", split = 'test', 
               n_batch=200, n_shot=2, n_query=8, pad_to_max_length = True, 
               max_seq_length = 128, tokenizer_name = 'bert-base-cased',
               overwrite_cache = False):
        """
        Args:
        root (str): root path of dataset.
        task_name (str): dataset name. 
        split (str): split of dataset used for testing, note not all dataset's test split has labels available
        n_batch (int): number of mini-batches per epoch. Default: 200
        n_shot (int): number of training (support) samples per category. 
            Default: 1
        n_query (int): number of validation (query) samples per category. 
            Default: 15
        pad_to_max_length (bool): "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        max_seq_length (int): "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        tokenizer_name (str): the name of tokenizer
        overwrite_cache (bool): Overwrite the cached preprocessed datasets or not.
        """
        super(FSDataset, self).__init__()

        self.task_name = task_name
        self.split = split
        self.n_batch = n_batch
        self.n_shot = n_shot
        self.n_query = n_query
        self.dataset = load_taskdata(task_name, split = split)

        self.n_shot = n_shot
        self.n_query = n_query

        self.padding = "max_length" if pad_to_max_length else False
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
        )
        self.overwrite_cache = overwrite_cache


        self.label = np.array(self.dataset['label'])
        self.label_list = self.dataset.unique("label")
        self.n_class = len(self.label_list)
        self.label_list = [i for i in range(self.n_class)]
        self.classlocs = tuple()
        # classlocs: a tuple with len() = n_class, show sample locations in each class
        for cat in self.label_list:
            self.classlocs += (np.argwhere(self.label == cat).reshape(-1),)
        
        self.dataset = self.dataset.map(
            self.process,
            batched=True,
            load_from_cache_file=not self.overwrite_cache,
            desc="Running tokenizer on dataset",
            remove_columns=self.dataset.column_names,
        )


    def __len__(self):
        return self.n_batch

    def process(self,examples):
        sentence1_key, sentence2_key = task_to_keys[self.task_name]
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result
    
    def __getitem__(self, index):
        dataset = self.dataset
        idx = []
        sq_idx = [] # store indices shot or query
        label_idx = [] # store indices indicates classes
        for i in self.label_list:
            # for one of the classes
            idx += list(np.random.choice(self.classlocs[i], self.n_shot + self.n_query))
            sq_idx += [0 for i  in range(self.n_shot)] + [1 for i  in range(self.n_query)]
            label_idx += [i for j in range(self.n_shot + self.n_query)]

        result = dataset.select(idx)

        data = result[:len(sq_idx)]
        for key, val in data.items():
            data[key] = torch.tensor(val)

        return torch.tensor(sq_idx), torch.tensor(label_idx), data # result is dataset contains len(sq_idx) rows , return __getitem__ set

        
class metaDataset(Dataset):
    def __init__(self, split = 'train', 
               n_batch=200, n_shot=2, n_query=8, pad_to_max_length = True, 
               max_seq_length = 128, tokenizer_name = 'bert-base-cased',
               overwrite_cache = False):
        """
        Args:
        root (str): root path of dataset.
        task_name (str): dataset name. 
        split (str): split of dataset used for testing, note not all dataset's test split has labels available
        n_batch (int): number of mini-batches per epoch. Default: 200
        n_shot (int): number of training (support) samples per category. 
            Default: 1
        n_query (int): number of validation (query) samples per category. 
            Default: 15
        pad_to_max_length (bool): "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        max_seq_length (int): "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        tokenizer_name (str): the name of tokenizer
        overwrite_cache (bool): Overwrite the cached preprocessed datasets or not.
        """
        super(metaDataset, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
        )
        self.overwrite_cache = overwrite_cache
        self.padding = "max_length" if pad_to_max_length else False
        self.max_seq_length = max_seq_length

        task_names = ['cola', 'sst2', 'qqp', 'mnli', 'qnli', 'rte', 'wnli', 'snli', 
            'trec', 'mpqa', 'cr', 'sst5', 'mr', 'subj']
        size_limit = 5000 if split == "train" else 1000

        tasks = {} # store truncated, preprocessed dataset
        self.labels = {} # store all the labels in each dataset
        for task_name in task_names:
            task_set = load_taskdata(task_name, split = split)
            task_set = task_set.select(
                random.sample(range(len(task_set)), min(size_limit, len(task_set)))
                )
            self.labels[task_name] = np.array(task_set['label'])
            tasks[task_name]= task_set.map(
                self.metaProcess(task_name),
                batched=True,
                load_from_cache_file=not overwrite_cache,
                # desc="Running tokenizer on dataset",
                remove_columns=task_set.column_names,
            )

        self.task_names = task_names
        self.tasks = tasks
         
        self.split = split
        self.n_batch = n_batch
        self.n_shot = n_shot
        self.n_query = n_query
  
        self.n_shot = n_shot
        self.n_query = n_query



    def __len__(self):
        return self.n_batch
    
    def metaProcess(self, task_name):
        def process(examples):
            sentence1_key, sentence2_key = task_to_keys[task_name]
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)
            return result
        
        return process
    
    def __getitem__(self, index):
        ### random dataset ###
        task_name = random.sample(self.task_names, 1)[0]
        dataset = self.tasks[task_name]

        label = self.labels[task_name]
        n_class = task_num_labels[task_name]
        label_list = [i for i in range(n_class)]
        classlocs = tuple()
        # classlocs: a tuple with len() = n_class, show sample locations in each class
        for cat in label_list:
            classlocs += (np.argwhere(label == cat).reshape(-1),)
        

        idx = []
        sq_idx = [] # store indices shot or query
        label_idx = [] # store indices indicates classes
        for i in label_list:
            # for one of the classes
            idx += list(np.random.choice(classlocs[i], self.n_shot + self.n_query))
            sq_idx += [0 for i  in range(self.n_shot)] + [1 for i  in range(self.n_query)]
            label_idx += [i for j in range(self.n_shot + self.n_query)]

        # print(idx)
        # print(sq_idx)
        # print(label_idx)
        result = dataset.select(idx)

        data = result[:len(sq_idx)]
        for key, val in data.items():
            data[key] = torch.tensor(val)

        return task_name, torch.tensor(sq_idx), torch.tensor(label_idx), data # result is dataset contains len(sq_idx) rows , return __getitem__ set

        



def load_taskdata(task_name: str, split):
    if task_name == 'mnli':
        if split == 'test':
            split = 'test_matched'
        elif split == 'validation':
            split ='validation_matched'

    if task_name in ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']:
        # GLUE tasks load directly
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            task_name,
            split = split
        )
    elif task_name == 'snli':
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            task_name,
            split = split
        )
    else:
        # Loading a dataset from your local files.
        # CSV files are needed.
        if split =='train':
            data_files = {"train": os.path.join(ROOT, task_name,"train.csv")}
        elif split == 'validation':
            data_files = {"validation":  os.path.join(ROOT, task_name,"validation.csv")}
        else:
            data_files = {"test": os.path.join(ROOT, task_name,"test.csv")}

        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            column_names=['label','sentence']
        )[split]
    if task_name == "snli":
        raw_datasets = raw_datasets.filter(lambda example: example["label"] != -1)
    if task_name == "cr" or task_name == "mpqa":
        raw_datasets = raw_datasets.filter(lambda example: example["sentence"] != None)

    return raw_datasets