#!/bin/bash

log_dir=${root_path}/log

exit_code=0

echo "*******paddlemix blip2 single card run_predict begin***********"
(CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_predict.py \
    --per_device_train_batch_size 64 \
    --num_train_epochs 1) 2>&1 | tee ${log_dir}/blip2_single_run_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到res.log
    echo "paddlemix blip2 single card run_predict run success" >> "${log_dir}/res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到res.log
    echo "paddlemix blip2 single card run_predict run fail" >> "${log_dir}/res.log"
fi
echo "*******paddlemix blip2 single card run_predict end***********"