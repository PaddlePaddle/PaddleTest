#!/bin/bash

log_dir=${root_path}/log

exit_code=0

echo "*******paddlemix eva02 eval begin***********"

MODEL_NAME="paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14"
DATA_PATH=${root_path}/dataset/ILSVRC2012_tiny
OUTPUT_DIR=./outputs

input_size=336
batch_size=128
num_workers=2

PRETRAIN_CKPT=None
(CUDA_VISIBLE_DEVICES=0 python run_eva02_finetune_eval.py \
  --do_eval \
  --model ${MODEL_NAME} \
  --pretrained_model_path ${PRETRAIN_CKPT} \
  --eval_data_path ${DATA_PATH}/val \
  --input_size ${input_size} \
  --per_device_eval_batch_size ${batch_size} \
  --dataloader_num_workers ${num_workers} \
  --output_dir ${OUTPUT_DIR} \
  --recompute True \
  --fp16 False) 2>&1 | tee ${log_dir}/run_mix_eva02_eval.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "paddlemix eva02 eval run success" >>"${log_dir}/ce_res.log"
else
  echo "paddlemix eva02 eval run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix eva02 eval end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
  exit 1
fi
