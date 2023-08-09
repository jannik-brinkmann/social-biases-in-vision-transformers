export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=
export TRANSFORMERS_CACHE=
export HF_DATASETS_CACHE=

python3 -m torch.distributed.launch \
    --nproc_per_node 6 ./../scripts/run_mim.py \
    --model_type "beit" \
    --cda "1-sided" \  # "2-sided" "imagenet"
    --output_dir ./outputs/ \
    --remove_unused_columns False \
    --label_names bool_masked_pos \
    --mask_ratio 0.4 \
    --do_train \
    --do_eval \
    --train_val_split 0.1 \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 100 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 100 \
    --seed 1337 \
    --push_to_hub False