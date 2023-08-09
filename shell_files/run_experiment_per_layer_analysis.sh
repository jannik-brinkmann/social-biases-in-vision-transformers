export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=
export TRANSFORMERS_CACHE=

for MODEL in 'microsoft/beit-base-patch16-224-pt22k' 'openai/imagegpt-small' 'facebook/dino-vitb16' 'facebook/vit-mae-base' 'facebook/vit-moco' 'facebook/vit-msn-base'; do
    for EXTRACTION_LAYER in 0 1 2 3 4 5 6 7 8 9 10 11; do
        python3 ./../run_ieat.py \
            --model_name_or_path ${MODEL} \
            --use_mean_pooling True \
            --extraction_layer ${EXTRACTION_LAYER}
    done;
done;