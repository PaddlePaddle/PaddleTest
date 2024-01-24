#!/bin/bash

log_dir=${root_path}/log

exit_code=0

# task_vqa
echo "*******paddlemix blip2 multi card run_eval_vqav2_zeroshot begin***********"
(fleetrun --gpus=0,1 paddlemix/examples/blip2/run_eval_vqav2_zeroshot.py \
    --per_device_train_batch_size 64 \
    --num_train_epochs 1) 2>&1 | tee ${log_dir}/blip2_multi_run_eval_vqav2_zeroshot.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix blip2 multi card run_eval_vqav2_zeroshot run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix blip2 multi card run_eval_vqav2_zeroshot run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix blip2 multi card run_eval_vqav2_zeroshot end***********"

# task_caption
echo "*******paddlemix blip2 multi card run_eval_caption begin***********"
(fleetrun --gpus=0,1 paddlemix/examples/blip2/run_eval_caption.py \
    --per_device_train_batch_size 64 \
    --num_train_epochs 1) 2>&1 | tee ${log_dir}/blip2_multi_run_eval_caption.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix blip2 multi card run_eval_caption run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix blip2 multi card run_eval_caption run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix blip2 multi card run_eval_caption end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
    exit 1
fi
