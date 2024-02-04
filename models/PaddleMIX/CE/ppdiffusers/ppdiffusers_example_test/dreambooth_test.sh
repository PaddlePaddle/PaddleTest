#!/bin/bash


export HF_ENDPOINT=https://hf-mirror.com
export FLAGS_conv_workspace_size_limit=4096
export INSTANCE_DIR="./dogs"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="dreambooth_danka"
export FLAG_FUSED_LINEAR=0
export FLAG_XFORMERS_ATTENTION_OP=auto

python -u train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --noise_offset 1 \
  --enable_xformers_memory_efficient_attention 
  
rm -rf ${OUTPUT_DIR}

# export OUTPUT_DIR="dreambooth_duoka"
python -u -m paddle.distributed.launch --gpus "0,1" train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --noise_offset 1 \
  --enable_xformers_memory_efficient_attention 

rm -rf ${OUTPUT_DIR}

# export OUTPUT_DIR="dreambooth_lora_danka"
python train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=50 \
  --learning_rate=1e-4 \
  --report_to="visualdl" \
  --gradient_checkpointing \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=5 \
  --lora_rank=4 \
  --checkpointing_steps 50 \
  --seed=0 \
  --noise_offset=1 \
  --train_text_encoder \
  --enable_xformers_memory_efficient_attention 

rm -rf ${OUTPUT_DIR}

# export OUTPUT_DIR="dreambooth_lora_duoka"
python -u -m paddle.distributed.launch --gpus "0,1" train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --gradient_checkpointing \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=50 \
  --learning_rate=1e-4 \
  --report_to="visualdl" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=5 \
  --checkpointing_steps 50 \
  --lora_rank=4 \
  --seed=0 \
  --noise_offset=1 \
  --train_text_encoder \
  --enable_xformers_memory_efficient_attention 

rm -rf ${OUTPUT_DIR}

export OUTPUT_DIR="dreambooth_lora_sdxl_danka"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

python train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="visualdl" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --checkpointing_steps=100 \
  --enable_xformers_memory_efficient_attention

rm -rf ${OUTPUT_DIR}
rm -rf ./dogs/
