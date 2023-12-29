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
    echo "paddlemix blip2 single card run_predict run success" >>"${log_dir}/res.log"
else
    echo "paddlemix blip2 single card run_predict run fail" >>"${log_dir}/res.log"
fi
echo "*******paddlemix blip2 single card run_predict end***********"

echo exit_code:${exit_code}
exit ${exit_code}
