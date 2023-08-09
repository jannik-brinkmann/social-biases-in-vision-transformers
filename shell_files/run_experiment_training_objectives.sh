export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=
export TRANSFORMERS_CACHE=

for MODEL in 'microsoft/beit-base-patch16-224-pt22k' 'openai/imagegpt-small' 'facebook/dino-vitb16' 'facebook/vit-mae-base' 'facebook/vit-moco' 'facebook/vit-msn-base'; do
    python3 ./../run_ieat.py \
        --model_name_or_path ${MODEL} \
        --use_mean_pooling True
done;