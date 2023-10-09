#!/bin/bash

echo "*******paddlemix visualglm predict begin***********"

(python run_predict.py \
    --pretrained_name_or_path "THUDM/visualglm-6b") 2>&1 | tee ${log_dir}/run_visualglm_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "paddlemix visualglm predict run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "paddlemix visualglm predict run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******paddlemix visualglm predict end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
  exit 1
fi