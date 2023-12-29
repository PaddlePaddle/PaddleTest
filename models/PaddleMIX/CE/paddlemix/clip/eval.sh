#!/bin/bash

log_dir=${root_path}/log

exit_code=0

echo "*******paddlemix clip eval begin***********"

MODEL_NAME="paddlemix/CLIP/Vit-L-14"

IN_1K_DIR=${root_path}/data/imagenet-val/

(python run_zero_shot_eval.py \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers=2 \
    --model ${MODEL_NAME} \
    --fp16 False \
    --pretrained_text_model Vit-L-14 \
    --classification_eval ${IN_1K_DIR} \
    --output_dir "output" \
    --disable_tqdm True) 2>&1 | tee ${log_dir}/run_mix_clip_eval.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix clip eval run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix clip eval run fail" >>"${log_dir}/ce_res.log"
fi

echo "*******paddlemix clip eval end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
    exit 1
fi
