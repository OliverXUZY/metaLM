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
        - Sample $m$ subdataset, sample 2 shots and 8 querys per classes points from each subdatasets.
        - Forward through encoder to get representations.
        - Compute loss nearest centroids method, backward gradient. 
        - Compute accuracy and save.

The evaluation is based on nearest-centroid. We do not learn linear head here.
Ideally, m = 5, now I took m = 1. Optimization details:

__epoch 1 to 50:__
```
optimizer: sgd
learning rate: 0.01
momentum: 1
weight decay: 0
```
**epoch 51 to 90:**
```
optimizer: sgd
learning rate: 1e-05
momentum: 1
weight decay: 0.0005
```

The accuracy for each epoch is averaged through all 200 batches.

### Result

|Original | Finetune|
|--|--|
|acc=0.54 +- 0.02 | acc=0.50 +- 0.00|

### Examples
* meta finetune backbone
```
python FStrain.py  \
    --model_name_or_path bert-base-cased \
    --gpu 0 \
    --save_epoch 10 \
    --lr 0.01 \
    --n_shot 2 \
    --n_query 8 \
    --num_batch 200 \
    --num_epoch 50
```
* test save backbone
```
python FStest.py --gpu 1 --output_dir save --n_shot 2 --n_query 8 --num_epoch 10
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
