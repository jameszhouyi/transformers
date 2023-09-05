echo "==========================================================================="
export http_proxy=http://proxy-dmz.intel.com:911
export https_proxy=http://proxy-dmz.intel.com:911
mkdir ./norwegian-gpt2
pip install datasets
pip install optax
pip install tokenizers
python clm_gpt2_train_tokenizer.py
pip install flax
python clm_gpt2_model_config.py
echo "==========================================================================="
source /local_dateset/basekit/46401/intel/oneapi/compiler/latest/env/vars.sh
source /local_dateset/basekit/46401/intel/oneapi/mkl/latest/env/vars.sh
export ZE_AFFINITY_MASK=0.0
python run_clm_flax.py \
    --output_dir="./norwegian-gpt2" \
    --model_type="gpt2" \
    --config_name="./norwegian-gpt2" \
    --tokenizer_name="./norwegian-gpt2" \
    --dataset_name="oscar" \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-3" --warmup_steps="1000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="1" \
    --logging_steps="500" \
    --save_steps="2500" \
    --eval_steps="2500" \
    --dtype="bfloat16"

