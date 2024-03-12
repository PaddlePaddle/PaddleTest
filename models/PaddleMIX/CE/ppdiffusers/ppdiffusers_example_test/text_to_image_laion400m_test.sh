#!/bin/bash

log_dir=${root_path}/examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

echo "*******text_to_image_laion400m single_train begin***********"
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
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file config/ldmbert.json \
    --unet_config_file config/unet.json \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 10 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased \
    --max_grad_norm -1 \
    --recompute True \
    --overwrite_output_dir \
    --benchmark True) 2>&1 | tee ${log_dir}/text_to_image_laion400m_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m single_train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m single_train end***********"

rm -rf ./laion400m_pretrain_output_trainer

echo "*******text_to_image_laion400m multi_train begin***********"
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
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file config/ldmbert.json \
    --unet_config_file config/unet.json \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 10 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased \
    --max_grad_norm -1 \
    --recompute True \
    --overwrite_output_dir \
    --benchmark True) 2>&1 | tee ${log_dir}/text_to_image_laion400m_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m multi_train end***********"

rm -rf ./laion400m_pretrain_output_trainer
rm -rf ./data/

echo exit_code:${exit_code}
exit ${exit_code}
