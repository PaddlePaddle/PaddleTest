#!/bin/bash

log_dir=${root_path}/examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0


echo "*******t2i_adapter single_train begin***********"
(python -u train_t2i_adapter_trainer.py \
    --do_train \
    --output_dir ./sd15_openpose_danka \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --max_steps 100 \
    --logging_steps 1 \
    --image_logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 1 \
    --seed 4096 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_list ./data_demo/train.openpose.filelist \
    --recompute True --use_ema False \
    --control_type raw \
    --data_format img2img \
    --use_paddle_conv_init False \
    --overwrite_output_dir \
    --timestep_sample_schedule cubic) 2>&1 | tee ${log_dir}/t2i_adapter_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "t2i_adapter single_train run success" >>"${log_dir}/ce_res.log"
else
    echo "t2i_adapter single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******t2i_adapter single_train end***********"

rm -rf ./sd15_openpose_danka

echo "*******t2i_adapter multi_train begin***********"
(python -u -m paddle.distributed.launch --gpus "0,1" train_t2i_adapter_trainer.py \
    --do_train \
    --output_dir ./sd15_openpose_duoka \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --max_steps 100 \
    --logging_steps 1 \
    --image_logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 1 \
    --seed 4096 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_list ./data_demo/train.openpose.filelist \
    --recompute True --use_ema False \
    --control_type raw \
    --data_format img2img \
    --use_paddle_conv_init False \
    --overwrite_output_dir \
    --timestep_sample_schedule cubic) 2>&1 | tee ${log_dir}/t2i_adapter_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "t2i_adapter multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "t2i_adapter multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******t2i_adapter multi_train end***********"

rm -rf ./sd15_openpose_duoka
rm -rf ./data/
rm -rf ./data_demo/

echo exit_code:${exit_code}
exit ${exit_code}
