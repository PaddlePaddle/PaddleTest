#!/bin/bash

log_dir=${root_path}/log

exit_code=0

# stage1
echo "*******paddlemix blip2 multi card run_pretrain_stage1 begin***********"
(fleetrun --gpus=0,1 paddlemix/examples/blip2/run_pretrain_stage1.py \
    --per_device_train_batch_size 64 \
    --num_train_epochs 1) | tee ${log_dir}/multi_run_pretrain_stage1.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "paddlemix blip2 multi card run_pretrain_stage2 run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "paddlemix blip2 multi card run_pretrain_stage2 run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******paddlemix blip2 multi card run_pretrain_stage1 end***********"

# stage2
echo "*******paddlemix blip2 multi card run_pretrain_stage2 begin***********"
(fleetrun --gpus=0,1 paddlemix/examples/blip2/run_pretrain_stage2.py \
    --per_device_train_batch_size 64 \
    --num_train_epochs 1) | tee ${log_dir}/multi_run_pretrain_stage2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "paddlemix blip2 multi card run_pretrain_stage2 run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "paddlemix blip2 multi card run_pretrain_stage2 run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******paddlemix blip2 multi card run_pretrain_stage2 end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
  exit 1
fi