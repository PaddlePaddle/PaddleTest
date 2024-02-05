#!/bin/bash

log_dir=${root_path}/examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

echo "*******autoencoder_vae single_train begin***********"
(python train_vae.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --ignore_keys decoder. \
    --vae_config_file config/vae.json \
    --freeze_encoder \
    --enable_xformers_memory_efficient_attention \
    --input_size 256 256 \
    --max_train_steps 100 \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --num_workers 4 \
    --logging_steps 25 \
    --save_steps 4000 \
    --image_logging_steps 25 \
    --disc_start 10 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512) 2>&1 | tee ${log_dir}/autoencoder_vae_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "autoencoder_vae single_train run success" >>"${log_dir}/ce_res.log"
else
    echo "autoencoder_vae single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******autoencoder_vae single_train end***********"

rm -rf ${work_path}/autoencoder_outputs/*

echo "*******autoencoder_vae multi_train begin***********"
(python -u -m paddle.distributed.launch --gpus "0,1" train_vae.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --ignore_keys decoder. \
    --vae_config_file config/vae.json \
    --freeze_encoder \
    --enable_xformers_memory_efficient_attention \
    --input_size 256 256 \
    --max_train_steps 100 \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --num_workers 4 \
    --logging_steps 25 \
    --save_steps 4000 \
    --image_logging_steps 25 \
    --disc_start 10 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512) 2>&1 | tee ${log_dir}/autoencoder_vae_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "autoencoder_vae multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "autoencoder_vae multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******autoencoder_vae multi_train infer end***********"

rm -rf ./autoencoder_outputs/*
rm -rf ./data/

echo exit_code:${exit_code}
exit ${exit_code}
