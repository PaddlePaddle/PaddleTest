#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/text_to_image_mscoco_uvit/
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

echo "*******text_to_image_mscoco_uvit train begin***********"
(bash train.sh) 2>&1 | tee ${log_dir}/text_to_image_mscoco_uvit_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_mscoco_uvit run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_mscoco_uvit run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_mscoco_uvit end***********"

echo exit_code:${exit_code}
exit ${exit_code}
