#!/bin/bash

log_dir=${root_path}/examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

echo "*******controlnet single_train begin***********"
(python -u train_txt2img_control_trainer.py \
    --do_train \
    --output_dir ./sd15_control_danka \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --sd_locked True \
    --max_steps 100 \
    --logging_steps 50 \
    --image_logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_path ./fill50k \
    --recompute True \
    --overwrite_output_dir) 2>&1 | tee ${log_dir}/controlnet_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "controlnet single_train run success" >>"${log_dir}/ce_res.log"
else
    echo "controlnet single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******controlnet single_train end***********"

rm -rf ./sd15_control_danka

echo "*******controlnet multi_train begin***********"
(python -u -m paddle.distributed.launch --gpus "0,1" train_txt2img_control_trainer.py \
    --do_train \
    --output_dir ./sd15_control_duoka \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --sd_locked True \
    --max_steps 100 \
    --logging_steps 50 \
    --image_logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_path ./fill50k \
    --recompute True \
    --overwrite_output_dir) 2>&1 | tee ${log_dir}/controlnet_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "controlnet multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "controlnet multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******controlnet multi_train infer end***********"

rm -rf ./sd15_control_duoka
rm -rf ./fill50k/

echo exit_code:${exit_code}
exit ${exit_code}
