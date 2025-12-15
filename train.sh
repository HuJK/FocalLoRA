#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate focallora
export CUDA_VISIBLE_DEVICES="0"
export PROJDIR="/data/local/hujk/FocalLoRA"
cd code

# Use the new dataset format

python _tuning.modified.py \
  --model_path "$PROJDIR/models/Llama-3.1-8B-Instruct/" \
  --json_path "$PROJDIR/data/all_combined.json" \
  --tune_path "$PROJDIR/data/focal_lora_dataset_train" \
  --output_dir "$PROJDIR/LoraAdapter/Llama-3.1-8B-Instruct_m_t10p_sink_orig3/" \
  --topk 10p \
  --epochs 10 \
  --batch_size 6 \
  --lr 5e-6 \
  --lambda_focus 0.1 \
#  --lora_path "$PROJDIR/LoraAdapter/Llama-3.1-8B-Instruct_modified_0.85/batch_0" \
#  --head_path "$PROJDIR/LoraAdapter/Llama-3.1-8B-Instruct/heads.py"

