export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=
export TRANSFORMERS_CACHE=

for MODEL in 'openai/imagegpt-small' 'openai/imagegpt-medium' 'openai/imagegpt-large'; do
    python3 ./../run_ieat.py \
        --model_name_or_path ${MODEL} \
        --use_mean_pooling True
done;

for MODEL in 'facebook/vit-mae-base' 'facebook/vit-mae-large' 'facebook/vit-mae-huge'; do
    python3 ./../run_ieat.py \
        --model_name_or_path ${MODEL} \
        --use_mean_pooling True
done;
