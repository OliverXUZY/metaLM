import os
import logging
from dataclasses import dataclass, field
from typing import Optional
import sys
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import transformers

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
)

import src
from src import (
    FSDataset, 
    load_taskdata,
    BertEncoder,
    FSCentroidClassifier,
    Model
)

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default = "bert-base-cased", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    num_epoch: int = field(
        default=1,
        metadata={"help": ("The number of epoches")},
    )

    gpu: int = field(
        default=0,
        metadata={"help": ("cuda id to use, 0 or 1")},
    )

    output_dir: str = field(
        default="save",
        metadata={"help": ("the output directory")},
    )
@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default='mrpc',
        metadata={"help": "The name of the task to test on."},
    )

    num_task: int = field(
        default=5,
        metadata={"help": ("The number of the categories(tasks) to train on one run")},
    )

    num_batch: int = field(
        default=200,
        metadata={"help": ("The number of the batches in each dataset, number of random draws")},
    )

    n_shot: int = field(
        default=2,
        metadata={"help": ("The number of shot images per-class in each draws")},
    )

    n_query: int = field(
        default=8,
        metadata={"help": ("The number of shot images per-class in each draws")},
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))

    model_args, data_args = parser.parse_args_into_dataclasses()

    if torch.cuda.is_available():
        DEVICE = torch.device(f'cuda:{model_args.gpu}')
    else:
        DEVICE =  torch.device("cpu")

    src.set_log_path(model_args.output_dir)

    src.ensure_path(model_args.output_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("my_log.log", mode='w')],
    )
    
    log_level = logging.INFO
    logger.setLevel(log_level)

    logger.info("load dataset")

    # sq_idx, label_idx, result = dataset[0]
    # print(sq_idx)
    # print(label_idx)
    # print(result.keys())

    ##### Dataset #####
    dataset = FSDataset(
        task_name=data_args.task_name, 
        split = 'test', 
        n_batch=data_args.num_batch,
        n_shot=data_args.n_shot, n_query=data_args.n_query,
        tokenizer_name = model_args.model_name_or_path
    )

    
    loader = DataLoader(dataset, batch_size=1)


    # sq_idx, label_idx, batch = next(iter(loader))
    # for key, val in batch.items():
    #     batch[key] = val[0]
    # sq_idx = sq_idx.view(-1)
    # label_idx = label_idx.view(-1)

    # print(sq_idx)
    # print(label_idx)
    # print(batch.keys(), batch['input_ids'].shape)

    ##### Model #####
    enc = BertEncoder.from_pretrained(model_args.model_name_or_path)
    clf = FSCentroidClassifier()

    model = Model(enc, clf, n_shot = data_args.n_shot)
    model = model.to(DEVICE)

    ##### Evaluation #####
    aves = {"ta": src.AverageMeter()}
    ta_lst = []
    model.eval()

    for epoch in range(1, model_args.num_epoch + 1):
        np.random.seed(epoch)
        with torch.no_grad():
            for (sq_idx, label_idx, batch) in tqdm(loader, desc='test', leave=False):
                for key, val in batch.items():
                    batch[key] = val[0].to(DEVICE)
                sq_idx = sq_idx.view(-1).to(DEVICE)
                label_idx = label_idx.view(-1).to(DEVICE)

                logits = model(sq_idx, label_idx, batch)

                y = label_idx[sq_idx == 1]

                acc = src.accuracy(logits, y)
                aves['ta'].update(acc)
                ta_lst.append(acc)

        src.log('[{}/{}]: acc={:.2f} +- {:.2f} (%)'.format(
            epoch, str(model_args.num_epoch), aves['ta'].item(), 
            src.mean_confidence_interval(ta_lst)), 
            filename = f'test_{dataset.n_shot}s{dataset.n_query}q_{model_args.model_name_or_path}.txt')




if __name__ == "__main__":
    main()





