#!/bin/bash


echo "*******paddlemix coca train begin***********"


MODEL_NAME="paddlemix/CoCa/coca_Vit-L-14/"
IN_1K_DIR=${root_path}/data/imagenet-val/

(python -m paddle.distributed.launch --nproc_per_node 8 run_pretrain_dist.py \
    --dataloader_num_workers=2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --model ${MODEL_NAME}  \
    --warmup_steps 100 \
    --learning_rate 5e-4 \
    --weight_decay 0.05 \
    --adam_beta1 0.9  \
    --adam_beta2 0.999  \
    --adam_epsilon 1e-8  \
    --max_grad_norm 5.0 \
    --num_train_epochs 1 \
    --tensor_parallel_degree 1 \
    --sharding_parallel_degree 8 \
    --sharding "stage2" \
    --bf16 False \
    --output_dir "./output" \
    --logging_steps 1 \
    --do_train \
    --disable_tqdm True \
    --save_steps 100 \
    --local_loss true \
    --gather_with_grad true \
    --pretrained_text_model ${MODEL_NAME} \
    --classification_eval ${IN_1K_DIR}) 2>&1 | tee ${log_dir}/run_mix_coca_train.log

tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "paddlemix coca train run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "paddlemix coca train run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******paddlemix coca train end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
  exit 1
fi