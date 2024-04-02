#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/class_conditional_image_generation/DiT/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

# 下载依赖和数据
bash prepare.sh

echo "*******class_conditional_image_generation/DiT train begin***********"
(bash train.sh) 2>&1 | tee ${log_dir}/class_conditional_image_generation_DiT_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "class_conditional_image_generation/DiT run success" >>"${log_dir}/ce_res.log"
else
    echo "class_conditional_image_generation/DiT run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******class_conditional_image_generation/DiT end***********"

echo exit_code:${exit_code}
exit ${exit_code}
