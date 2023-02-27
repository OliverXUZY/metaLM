python FStest.py \
    --model_name_or_path save/save_2s8q_optimadamW \
    --tokenizer_name save/save_2s8q_optimadamW \
    --gpu 1 \
    --output_dir save \
    --n_shot 2 \
    --n_query 8 \
    --num_epoch 10