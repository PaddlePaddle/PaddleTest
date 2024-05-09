#!/bin/bash

log_dir=${root_path}/examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

echo "*******stable_diffusion single_train begin***********"
(python -u train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 100 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 5 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 10 \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --recompute True \
    --overwrite_output_dir \
    --benchmark True \
    --fp16_opt_level O2) 2>&1 | tee ${log_dir}/stable_diffusion_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "stable_diffusion single_train run success" >>"${log_dir}/ce_res.log"
else
    echo "stable_diffusion single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******stable_diffusion single_train end***********"

rm -rf ./laion400m_pretrain_output_trainer

# CUDA version needs to be greater than 11.7.
echo "*******stable_diffusion multi_train begin***********"
(python -u -m paddle.distributed.launch --gpus "0,1" train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 100 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 5 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 10 \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --recompute True \
    --overwrite_output_dir \
    --benchmark True \
    --fp16_opt_level O2) 2>&1 | tee ${log_dir}/stable_diffusion_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "stable_diffusion multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "stable_diffusion multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******stable_diffusion multi_train end***********"

rm -rf ./laion400m_pretrain_output_trainer
rm -rf ./data/

echo exit_code:${exit_code}
exit ${exit_code}
