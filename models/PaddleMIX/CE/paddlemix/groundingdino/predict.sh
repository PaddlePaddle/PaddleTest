#!/bin/bash

log_dir=${root_path}/log

exit_code=0

echo "*******paddlemix groundingdino predict begin***********"

(python run_predict.py \
  --input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
  --prompt "bus") 2>&1 | tee ${log_dir}/run_groundingdino_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "paddlemix groundingdino predict run success" >>"${log_dir}/ce_res.log"
else
  echo "paddlemix groundingdino predict run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix groundingdino predict end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
  exit 1
fi
