#!/bin/bash

rm -rf ./laion400m_pretrain_output_trainer
export FLAGS_conv_workspace_size_limit=4096
# 是否开启ema
export FLAG_USE_EMA=0
# 是否开启recompute
export FLAG_RECOMPUTE=1
# 是否开启xformers
export FLAG_XFORMERS=1
python -u -m paddle.distributed.launch --gpus "0,1" train_txt2img_laion400m_trainer.py \
  --do_train \
  --output_dir ./laion400m_pretrain_output_trainer \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --max_steps 100 \
  --lr_scheduler_type "constant" \
  --warmup_steps 0 \
  --image_logging_steps 100 \
  --logging_steps 10 \
  --resolution 256 \
  --save_steps 100 \
  --save_total_limit 20 \
  --seed 23 \
  --dataloader_num_workers 8 \
  --pretrained_model_name_or_path ./CompVis-stable-diffusion-v1-4-paddle-init \
  --file_list ./data/filelist/train.filelist.list \
  --model_max_length 77 \
  --max_grad_norm -1 \
  --disable_tqdm True
# --bf16 True

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
