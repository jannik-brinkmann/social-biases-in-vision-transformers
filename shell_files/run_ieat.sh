export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=
export TRANSFORMERS_CACHE=

for MODEL in ''; do
    python3 ./../run_ieat.py \
        --model_name_or_path ${MODEL} \
        --use_mean_pooling True
done;