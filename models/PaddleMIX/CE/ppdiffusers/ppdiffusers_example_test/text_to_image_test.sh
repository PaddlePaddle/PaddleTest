#!/bin/bash

log_dir=${root_path}/examples_log

if [ ! -d "$log_dir" ]; then
  mkdir -p "$log_dir"
fi

exit_code=0

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export OUTPUT_DIR="sd-pokemon-model"

echo "*******text_to_image single_train begin***********"
(python -u train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --debug \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=50 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --noise_offset=1 \
  --output_dir=${OUTPUT_DIR}) 2>&1 | tee ${log_dir}/text_to_image_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "text_to_image single_train run success" >>"${log_dir}/ce_res.log"
else
  echo "text_to_image single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image single_train end***********"

rm -rf ${OUTPUT_DIR}

export OUTPUT_DIR="sd-pokemon-model-duoka"
echo "*******text_to_image multi_train begin***********"
(python -u -m paddle.distributed.launch --gpus "0,1" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --debug \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=50 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --noise_offset=1 \
  --output_dir=${OUTPUT_DIR}) 2>&1 | tee ${log_dir}/text_to_image_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "text_to_image multi_train run success" >>"${log_dir}/ce_res.log"
else
  echo "text_to_image multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image multi_train end***********"

rm -rf ${OUTPUT_DIR}

export OUTPUT_DIR="sd-pokemon-model-lora"
echo "*******text_to_image lora_single_train begin***********"
(python train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=50 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --debug \
  --gradient_checkpointing \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=visualdl \
  --checkpointing_steps=50 \
  --validation_prompt="Totoro" \
  --lora_rank=4 \
  --seed=1337 \
  --noise_offset=1 \
  --validation_epochs 1 \
  --enable_xformers_memory_efficient_attention) 2>&1 | tee ${log_dir}/text_to_image_lora_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "text_to_image lora_single_train run success" >>"${log_dir}/ce_res.log"
else
  echo "text_to_image lora_single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image lora_single_train end***********"

rm -rf ${OUTPUT_DIR}

export OUTPUT_DIR="sd-pokemon-model-lora-duoka"
echo "*******text_to_image lora_multi_train begin***********"
(python -u -m paddle.distributed.launch --gpus "0,1" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=50 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --debug \
  --gradient_checkpointing \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=visualdl \
  --checkpointing_steps=50 \
  --validation_prompt="Totoro" \
  --lora_rank=4 \
  --seed=1337 \
  --noise_offset=1 \
  --validation_epochs 1 \
  --enable_xformers_memory_efficient_attention) 2>&1 | tee ${log_dir}/text_to_image_lora_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "text_to_image lora_multi_train run success" >>"${log_dir}/ce_res.log"
else
  echo "text_to_image lora_multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image lora_multi_train end***********"

rm -rf ${OUTPUT_DIR}

# sdxl
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
RESOLUTION=768
# export OUTPUT_DIR="sd-pokemon-model-sdxl"

# python -u train_text_to_image_sdxl.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --pretrained_vae_model_name_or_path=$VAE_NAME \
#   --dataset_name=$DATASET_NAME \
#   --enable_xformers_memory_efficient_attention \
#   --resolution=${RESOLUTION} --center_crop --random_flip \
#   --proportion_empty_prompts=0.2 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 --gradient_checkpointing \
#   --max_train_steps=50 \
#   --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --mixed_precision="fp16" \
#   --report_to="visualdl" \
#   --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 1 \
#   --checkpointing_steps=20 \
#   --output_dir=${OUTPUT_DIR}

# export OUTPUT_DIR="sd-pokemon-model-sdxl-duoka"
# python -u -m paddle.distributed.launch --gpus "0,1" train_text_to_image_sdxl.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --pretrained_vae_model_name_or_path=$VAE_NAME \
#   --dataset_name=$DATASET_NAME \
#   --enable_xformers_memory_efficient_attention \
#   --resolution=${RESOLUTION} --center_crop --random_flip \
#   --proportion_empty_prompts=0.2 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 --gradient_checkpointing \
#   --max_train_steps=50 \
#   --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --mixed_precision="fp16" \
#   --report_to="visualdl" \
#   --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 1 \
#   --checkpointing_steps=20 \
#   --output_dir=${OUTPUT_DIR}

export OUTPUT_DIR="sd-pokemon-model-lora-sdxl"
echo "*******text_to_image lora_sdxl_single_train begin***********"
(python -u train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=${RESOLUTION} --random_flip \
  --train_batch_size=1 \
  --max_train_steps=50 --checkpointing_steps=20 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir=${OUTPUT_DIR} --validation_epochs 1 \
  --train_text_encoder \
  --validation_prompt="cute dragon creature" --report_to="visualdl") 2>&1 | tee ${log_dir}/text_to_image_lora_sdxl_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "text_to_image lora_sdxl_single_train run success" >>"${log_dir}/ce_res.log"
else
  echo "text_to_image lora_sdxl_single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image lora_sdxl_single_train end***********"

rm -rf ${OUTPUT_DIR}

export OUTPUT_DIR="sd-pokemon-model-lora-sdxl-duoka"
echo "*******text_to_image lora_sdxl_multi_train begin***********"
(python -u -m paddle.distributed.launch --gpus "0,1" train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=${RESOLUTION} --random_flip \
  --train_batch_size=1 \
  --max_train_steps=50 --checkpointing_steps=20 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir=${OUTPUT_DIR} --validation_epochs 1 \
  --train_text_encoder \
  --validation_prompt="cute dragon creature" --report_to="visualdl") 2>&1 | tee ${log_dir}/text_to_image_lora_sdxl_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "text_to_image lora_sdxl_multi_train run success" >>"${log_dir}/ce_res.log"
else
  echo "text_to_image lora_sdxl_multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image lora_sdxl_multi_train end***********"

rm -rf ${OUTPUT_DIR}
rm -rf ./sd-pokemon-model/*

echo exit_code:${exit_code}
exit ${exit_code}
