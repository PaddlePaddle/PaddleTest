#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/text_to_image/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

# 单机训练
export http_proxy=${proxy}
export https_proxy=${proxy}
echo "*******text_to_image singe_train begin***********"
(bash single_train.sh) 2>&1 | tee ${log_dir}/text_to_image_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image singe_train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image singe_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image singe_train end***********"
unset http_proxy
unset https_proxy

# 单机训练的结果进行推理
echo "******text_to_image singe infer begin***********"
(python infer.py 2>&1) | tee ${log_dir}/text_to_image_single_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image single_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image single_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image singe infer end***********"

export http_proxy=${proxy}
export https_proxy=${proxy}
# 多机训练
echo "*******text_to_image muti_train begin***********"
(bash multi_train.sh) 2>&1 | tee ${log_dir}/text_to_image_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image multi_train end***********"
unset http_proxy
unset https_proxy

# 多机训练的结果进行推理
echo "*******text_to_image multi infer begin***********"
(python infer.py) 2>&1 | tee ${log_dir}/text_to_image_multi_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image multi_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image multi_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image multi infer end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}
