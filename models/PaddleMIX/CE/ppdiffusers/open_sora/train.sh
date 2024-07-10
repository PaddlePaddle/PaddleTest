# 单机多卡训练

ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
python -u -m paddle.distributed.launch --gpus "0,1,2,3" scripts/trainer_opensora.py \
    --do_train \
    --output_dir ./exp_output \
    --save_strategy 'steps' \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2.0e-5 \
    --max_steps 1000 \
    --seed 42 \
    --sharding "stage1" \
    --report_to all \
    --fp16 True \
    --fp16_opt_level O1

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
