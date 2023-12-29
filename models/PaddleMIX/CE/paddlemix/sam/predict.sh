#!/bin/bash

log_dir=${root_path}/log

exit_code=0

echo "*******paddlemix sam predict begin***********"

#box
(python run_predict.py \
    --input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
    --box_prompt 112 118 513 382 \
    --input_type boxs) 2>&1 | tee ${log_dir}/run_sam_box_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix sam box predict run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix sam box predict run fail" >>"${log_dir}/ce_res.log"
fi

#points
(python run_predict.py \
    --input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
    --points_prompt 362 250 \
    --input_type points) 2>&1 | tee ${log_dir}/run_sam_point_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix sam point predict run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix sam point predict run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix sam predict end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
    exit 1
fi
