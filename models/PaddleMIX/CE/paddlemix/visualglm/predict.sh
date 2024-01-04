#!/bin/bash

log_dir=${root_path}/log

exit_code=0

echo "*******paddlemix visualglm predict begin***********"

(python run_predict.py \
    --pretrained_name_or_path "THUDM/visualglm-6b" \
    --image_path "https://paddlenlp.bj.bcebos.com/data/images/mugs.png") 2>&1 | tee ${log_dir}/run_visualglm_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix visualglm predict run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix visualglm predict run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix visualglm predict end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
    exit 1
fi
