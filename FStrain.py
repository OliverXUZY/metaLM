import os
import logging
from dataclasses import dataclass, field
from typing import Optional
import sys
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import transformers

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
)

import src
from src import (
    metaDataset, 
    load_taskdata,
    BertEncoder,
    FSCentroidClassifier,
    Model,
    optimizers,
)

logger = logging.getLogger(__name__)


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
        metadata={"help": ("The cuda id to use, 0 or 1")},
    )

    output_dir: str = field(
        default="save",
        metadata={"help": ("The output directory")},
    )

    save_epoch: int = field(
        default=None,
        metadata={"help": ("Save model every certain epochs")},
    )

    optimizer: str = field(
        default="sgd",
        metadata={"help": ("The optimizer")},
    )

    lr: float = field(
        default=5e-5,
        metadata={"help": ("The learning rate")},
    )

    momentum: float = field(
        default=1,
        metadata={"help": ("The momentum to use")},
    )

    weight_decay: float = field(
        default=0,
        metadata={"help": ("The weight decay to se")},
    )

    start_epoch_from: int = field(
        default=0,
        metadata={"help": ("resume from checkpoint from epoch \{\}")},
    )




@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    num_datasets: int = field(
        default=1,
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
    
    ckpt_name = 'save_{}s{}q_{}_batch{}'.format(
        data_args.n_shot, data_args.n_query,
        model_args.optimizer, data_args.num_batch
        )
    ckpt_path = os.path.join(model_args.output_dir, ckpt_name)
    src.ensure_path(ckpt_path)
    src.set_log_path(ckpt_path)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("my_log.log", mode='w')],
    )
    
    log_level = logging.INFO
    # logger.setLevel(log_level)
    # logger.setLevel(logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        )

    logger.info("load dataset")
    
    ##### Dataset #####
    train_dataset = metaDataset(
        split = 'train', 
        n_batch=data_args.num_batch * data_args.num_datasets,
        n_shot=data_args.n_shot, n_query=data_args.n_query,
        tokenizer = tokenizer
    )

    # name, sq_idx, label_idx, result = train_dataset[0]

    # print(name)
    # print(sq_idx)
    # print(label_idx)
    # print(result.keys())

    train_loader = DataLoader(train_dataset, batch_size=1)


    # name, sq_idx, label_idx, batch = next(iter(train_loader))
    # for key, val in batch.items():
    #     batch[key] = val[0]
    # sq_idx = sq_idx.view(-1)
    # label_idx = label_idx.view(-1)
    # name = name[0]

    # print(name)
    # print(sq_idx)
    # print(label_idx)
    # print(batch.keys(), batch['input_ids'].shape)
###############################################################################################################################################################################
######################################################################################################################################
    val_dataset = metaDataset(
        split = 'validation', 
        n_batch=data_args.num_batch * data_args.num_datasets,
        n_shot=data_args.n_shot, n_query=data_args.n_query,
        tokenizer = tokenizer
    )

    val_loader = DataLoader(val_dataset, batch_size=1)

    ##### Model and Optimizer#####
    enc = BertEncoder.from_pretrained(model_args.model_name_or_path)
    clf = FSCentroidClassifier()

    model = Model(enc, clf, n_shot = data_args.n_shot)
    model = model.to(DEVICE)

    src.log('num params: {}'.format(src.count_params(model)))

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": model_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optimizers.make(model_args.optimizer, optimizer_grouped_parameters, 
                                lr = model_args.lr, momentum = model_args.momentum, 
                                )
    
    ### log opt info to file
    start_epoch_from = model_args.start_epoch_from
    with open(os.path.join(ckpt_path, "optim_config.txt"), "a") as f:
        f.write("epoch {} to {}\n".format(start_epoch_from + 1, start_epoch_from + model_args.num_epoch))
        f.write("optimizer: {}\n".format(model_args.optimizer))
        f.write("learning rate: {}\n".format(model_args.lr))
        f.write("momentum: {}\n".format(model_args.momentum))
        f.write("weight decay: {}\n".format(model_args.weight_decay))


    ##### Training and evaluation #####
    src.log("***** Running training *****")
    src.log(f"  Num Epochs = {model_args.num_epoch}")
    src.log(f"  Num batches(draws) = {len(train_dataset)/data_args.num_datasets}")
    src.log(f"  Instantaneous batch size per device = {data_args.n_shot}-shot, {data_args.n_query}-query")
    # src.log(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    timer_elapsed, timer_epoch = src.Timer(), src.Timer()


  
    loss_func = nn.CrossEntropyLoss().to(DEVICE)

    aves_keys = ['tl', 'ta', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []
    
    start_epoch = 1
    max_va = 0.

    for epoch in range(1, model_args.num_epoch + 1):
        np.random.seed(epoch)
        timer_epoch.start()
        aves = {k: src.AverageMeter() for k in aves_keys}

        model.train() 
        dataset_list = []
        for step, (name, sq_idx, label_idx, batch) in enumerate(tqdm(train_loader, desc='train', leave=False)):
            if name in dataset_list:
                continue
            dataset_list.append(name)
            for key, val in batch.items():
                batch[key] = val[0].to(DEVICE)
            sq_idx = sq_idx.view(-1).to(DEVICE)
            label_idx = label_idx.view(-1).to(DEVICE)
            name = name[0]

            logits = model(sq_idx, label_idx, batch)
            y = label_idx[sq_idx == 1]
            loss = loss_func(logits, y)

            ## 5 datasets then update gradient once
            loss = loss / data_args.num_datasets
            loss.backward()
            

            acc = src.accuracy(logits, y)

            aves['tl'].update(loss.item())
            aves['ta'].update(acc)

            if len(dataset_list) == data_args.num_datasets or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                dataset_list = []

            # src.log(f"epoch: {epoch} --train loss: {aves['tl'].item()}, acc: {aves['ta'].item()}")

        
        # meta-val
        model.eval()
        np.random.seed(0)
        with torch.no_grad():
            for (name, sq_idx, label_idx, batch) in tqdm(val_loader, desc='val', leave=False):
                for key, val in batch.items():
                    batch[key] = val[0].to(DEVICE)
                sq_idx = sq_idx.view(-1).to(DEVICE)
                label_idx = label_idx.view(-1).to(DEVICE)
                name = name[0]

                logits = model(sq_idx, label_idx, batch)
                y = label_idx[sq_idx == 1]
                loss = loss_func(logits, y)

                acc = src.accuracy(logits, y)

                aves['vl'].update(loss.item())
                aves['va'].update(acc)
        
        # src.log(f"epoch: {epoch} --val loss: {aves['vl'].item()}, acc: {aves['va'].item()}")

        for k, avg in aves.items():
            aves[k] = avg.item()
            trlog[k].append(aves[k])

        t_epoch = src.time_str(timer_epoch.end())
        t_elapsed = src.time_str(timer_elapsed.end())
        t_estimate = src.time_str(timer_elapsed.end() / 
            (epoch - start_epoch + 1) * (model_args.num_epoch - start_epoch + 1))
        
        # formats output
        log_str = '[{}/{}] train {:.4f}(C)|{:.2f}'.format(
            str(start_epoch_from + epoch), str(start_epoch_from + model_args.num_epoch), aves['tl'], aves['ta'])
        
        log_str += ', val {:.4f}(C)|{:.2f}'.format(aves['vl'], aves['va'])

        log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
        src.log(log_str)

        
        # saves model and meta-data

        if aves['va'] > max_va:
            max_va = aves['va']
            model.enc.save_pretrained(os.path.join(ckpt_path, "max-va"))
            tokenizer.save_pretrained(os.path.join(ckpt_path, "max-va"))

        
        if model_args.save_epoch and epoch % model_args.save_epoch == 0:
            model.enc.save_pretrained(os.path.join(ckpt_path, "checkpoint-{}".format(start_epoch_from + model_args.save_epoch)))
            tokenizer.save_pretrained(os.path.join(ckpt_path, "checkpoint-{}".format(start_epoch_from + model_args.save_epoch)))

        model.enc.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)


        

if __name__ == "__main__":
    main()

