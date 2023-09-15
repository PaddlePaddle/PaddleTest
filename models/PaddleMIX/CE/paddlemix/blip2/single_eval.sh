#!/bin/bash

log_dir=${root_path}/log

exit_code=0

# task_vqa
echo "*******paddlemix blip2 single card run_eval_vqav2_zeroshot begin***********"
(CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_eval_vqav2_zeroshot.py \
    --per_device_train_batch_size 64 \
    --num_train_epochs 1) | tee ${log_dir}/single_run_eval_vqav2_zeroshot.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "paddlemix blip2 single card run_eval_vqav2_zeroshot run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "paddlemix blip2 single card run_eval_vqav2_zeroshot run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******paddlemix blip2 single card run_eval_vqav2_zeroshot end***********"


# task_caption
echo "*******paddlemix blip2 single card run_eval_caption begin***********"
(CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_eval_caption.py \
    --per_device_train_batch_size 64 \
    --num_train_epochs 1) | tee ${log_dir}/single_run_eval_caption.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "paddlemix blip2 single card run_eval_caption run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "paddlemix blip2 single card run_eval_caption run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******paddlemix blip2 single card run_eval_caption end***********"