#!/bin/bash

echo "*******paddlemix evaclip eval begin***********"

MODEL_NAME="paddlemix/EVA/EVA02-CLIP-L-14"

IN_1K_DIR=${root_path}/data/imagenet-val/

(python run_zero_shot_eval.py \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers=2 \
    --model ${MODEL_NAME}  \
    --fp16 False \
    --pretrained_text_model EVA02-CLIP-L-14 \
    --classification_eval ${IN_1K_DIR} \
    --output_dir "output" \
    --disable_tqdm True) 2>&1 | tee ${log_dir}/run_mix_evaclip_eval.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "paddlemix evaclip eval run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "paddlemix evaclip eval run fail" >> "${log_dir}/ce_res.log"
fi

echo "*******paddlemix evaclip eval end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
  exit 1
fi