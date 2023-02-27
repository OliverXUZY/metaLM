# metaLM

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
