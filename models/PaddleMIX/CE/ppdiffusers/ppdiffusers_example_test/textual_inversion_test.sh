#!/bin/bash

log_dir=${root_path}/examples_log

if [ ! -d "$log_dir" ]; then
  mkdir -p "$log_dir"
fi

exit_code=0

export DATA_DIR="cat-toy"
export MODEL_NAME=runwayml/stable-diffusion-v1-5

export OUTPUT_DIR="textual_inversion_cat"
echo "*******textual_inversion single_train begin***********"
(python -u train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed 42 \
  --save_steps 50 \
  --gradient_checkpointing \
  --validation_prompt "A <cat-toy> backpack" \
  --validation_epochs 1 \
  --noise_offset 1 \
  --output_dir=${OUTPUT_DIR}) 2>&1 | tee ${log_dir}/textual_inversion_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "textual_inversion single_train run success" >>"${log_dir}/ce_res.log"
else
  echo "textual_inversion single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******textual_inversion single_train end***********"

rm -rf ${OUTPUT_DIR}

export OUTPUT_DIR="textual_inversion_cat_duoka"
echo "*******textual_inversion multi_train begin***********"
(python -u -m paddle.distributed.launch --gpus "0,1" train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed 42 \
  --save_steps 50 \
  --gradient_checkpointing \
  --validation_prompt "A <cat-toy> backpack" \
  --validation_epochs 1 \
  --noise_offset 1 \
  --output_dir=${OUTPUT_DIR}) 2>&1 | tee ${log_dir}/textual_inversion_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "textual_inversion multi_train run success" >>"${log_dir}/ce_res.log"
else
  echo "textual_inversion multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******textual_inversion multi_train end***********"

rm -rf ${OUTPUT_DIR}

echo exit_code:${exit_code}
exit ${exit_code}
