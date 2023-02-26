python FStrain.py  \
    --model_name_or_path bert-base-cased \
    --gpu 0 \
    --save_epoch 10 \
    --lr 0.01 \
    --n_shot 2 \
    --n_query 8 \
    --num_batch 200 \
    --num_epoch 20









# python FStest.py --gpu 1 --output_dir save --n_shot 2 --n_query 8 --num_epoch 5