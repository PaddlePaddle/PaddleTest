#!/bin/bash

log_dir=${root_path}/examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

MAX_ITER=50
MODEL_NAME_OR_PATH="runwayml/stable-diffusion-v1-5"
IS_SDXL=False
RESOLUTION=512

echo "*******lcm_trainer lora_single_train begin***********"
(python train_lcm.py \
    --do_train \
    --output_dir "lcm_lora_outputs" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 1 \
    --resolution ${RESOLUTION} \
    --save_steps 25 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path ${MODEL_NAME_OR_PATH} \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm 1 \
    --disable_tqdm True \
    --overwrite_output_dir \
    --recompute True \
    --loss_type "huber" \
    --lora_rank 64 \
    --is_sdxl ${IS_SDXL} \
    --is_lora True \
    --overwrite_output_dir \
    --fp16 True \
    --fp16_opt_level O2) 2>&1 | tee ${log_dir}/lcm_trainer_lora_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "lcm_trainer lora_single_train run success" >>"${log_dir}/ce_res.log"
else
    echo "lcm_trainer lora_single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******lcm_trainer lora_single_train end***********"

rm -rf ./lcm_lora_outputs

echo "*******lcm_trainer lora_multi_train begin***********"
(python -u -m paddle.distributed.launch --gpus "0,1" train_lcm.py \
    --do_train \
    --output_dir "lcm_lora_n1c2_outputs" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 1 \
    --resolution ${RESOLUTION} \
    --save_steps 25 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path ${MODEL_NAME_OR_PATH} \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm 1 \
    --disable_tqdm True \
    --overwrite_output_dir \
    --recompute True \
    --loss_type "huber" \
    --lora_rank 64 \
    --is_sdxl ${IS_SDXL} \
    --is_lora True \
    --overwrite_output_dir \
    --fp16 True \
    --fp16_opt_level O2) 2>&1 | tee ${log_dir}/lcm_trainer_lora_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "lcm_trainer lora_multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "lcm_trainer lora_multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******lcm_trainer lora_multi_train end***********"

rm -rf ./lcm_lora_n1c2_outputs

MAX_ITER=50
MODEL_NAME_OR_PATH="stabilityai/stable-diffusion-xl-base-1.0"
IS_SDXL=True
RESOLUTION=512

echo "*******lcm_trainer sdxl_lora_single_train begin***********"
(python train_lcm.py \
    --do_train \
    --output_dir "lcm_sdxl_lora_outputs" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 2000000 \
    --logging_steps 1 \
    --resolution ${RESOLUTION} \
    --save_steps 25 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path ${MODEL_NAME_OR_PATH} \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm 1 \
    --disable_tqdm True \
    --overwrite_output_dir \
    --recompute True \
    --loss_type "huber" \
    --lora_rank 64 \
    --is_sdxl ${IS_SDXL} \
    --is_lora True \
    --overwrite_output_dir) 2>&1 | tee ${log_dir}/lcm_trainer_sdxl_lora_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "lcm_trainer sdxl_lora_single_train run success" >>"${log_dir}/ce_res.log"
else
    echo "lcm_trainer sdxl_lora_single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******lcm_trainer sdxl_lora_single_train end***********"

rm -rf ./lcm_sdxl_lora_outputs

echo "*******lcm_trainer sdxl_lora_multi_train begin***********"
(python -u -m paddle.distributed.launch --gpus "0,1" train_lcm.py \
    --do_train \
    --output_dir "lcm_sdxl_lora_n1c2_outputs" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 2000000 \
    --logging_steps 1 \
    --resolution ${RESOLUTION} \
    --save_steps 25 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path ${MODEL_NAME_OR_PATH} \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm 1 \
    --disable_tqdm True \
    --overwrite_output_dir \
    --recompute True \
    --loss_type "huber" \
    --lora_rank 64 \
    --is_sdxl ${IS_SDXL} \
    --is_lora True \
    --overwrite_output_dir) 2>&1 | tee ${log_dir}/lcm_trainer_sdxl_lora_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "lcm_trainer sdxl_lora_multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "lcm_trainer sdxl_lora_multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******lcm_trainer sdxl_lora_multi_train end***********"

rm -rf ./lcm_sdxl_lora_n1c2_outputs
rm -rf ./data/

echo exit_code:${exit_code}
exit ${exit_code}
