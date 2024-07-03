ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
python -u -m paddle.distributed.launch --gpus "0,1,2,3" scripts/trainer_stage1.py \
    --do_train \
    --output_dir ./exp_output/stage1 \
    --save_strategy 'steps' \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --weight_decay 1.0e-2 \
    --max_steps 1000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 1 \
    --seed 42 \
    --report_to all \
    --sharding "stage1" \
    --fp16 True \
    --fp16_opt_level O2

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
