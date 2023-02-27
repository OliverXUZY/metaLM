# metaLM

### Datasets
GLUE like datasets: 
- train dataset: cola, sst2, qqp, mnli, qnli, rte, wnli, snli, 
            trec, mpqa, cr, sst5, mr, subj. The datasets contains binary classes or multi-classes.
- test dataset: mrpc
### Algorithms:
1. Sub-sample subdatasets with 5000 samples from each datasets.
2. For epoch = 1, ..., n_epoch
    1. For batch = 1, ..., 200:
        - Sample $m$ subdataset, sample 2 shot smaples and 8 query samples per classes from each subdatasets.
        - Forward through encoder to get representations.
        - Compute loss nearest centroids method, backward gradient. 
        - Compute accuracy and save.

The evaluation is based on nearest-centroid. We do not learn linear head here.
Ideally, m = 5, now I took m = 1. Optimization details:

__epoch 1 to 20:__
```
optimizer: adamW
learning rate: 2e-05
momentum: 1
weight decay: 5e-4
```
The accuracy for each epoch is averaged through all 200 batches.

### Result

|Original | Finetune|
|--|--|
|0.53 +- 0.01 | 0.60 +- 0.01|

### Examples
* meta finetune backbone
```
python FStrain.py  \
    --model_name_or_path bert-base-cased \
    --gpu 0 \
    --save_epoch 10 \
    --optimizer adamW \
    --lr 2e-5 \
    --weight_decay 5e-4 \
    --n_shot 2 \
    --n_query 8 \
    --num_batch 200 \
    --num_epoch 20 \
    --start_epoch_from 0
```
* test save backbone
```
python FStest.py \
    --model_name_or_path save/save_2s8q_adamW \
    --tokenizer_name save/save_2s8q_adamW \
    --gpu 1 \
    --output_dir save \
    --n_shot 2 \
    --n_query 8 \
    --num_epoch 10
```


### Local data directory
```
materials/
├── cr
│   ├── custrev.all
│   ├── process.py
│   ├── test.csv
│   ├── train.csv
│   └── validation.csv
├── dataset.xlsx
├── mpqa
│   ├── mpqa.all
│   ├── process.py
│   ├── test.csv
│   ├── train.csv
│   └── validation.csv
├── mr
│   ├── mr.all
│   ├── process.py
│   ├── test.csv
│   ├── train.csv
│   └── validation.csv
├── sst5
│   ├── process.py
│   ├── stsa.fine.dev
│   ├── stsa.fine.test
│   ├── stsa.fine.train
│   ├── test.csv
│   ├── train.csv
│   └── validation.csv
├── subj
│   ├── process.py
│   ├── subj.all
│   ├── test.csv
│   ├── train.csv
│   └── validation.csv
└── trec
    ├── TREC.test.all
    ├── TREC.train.all
    ├── process.py
    ├── test.csv
    ├── train.csv
    └── validation.csv
```
