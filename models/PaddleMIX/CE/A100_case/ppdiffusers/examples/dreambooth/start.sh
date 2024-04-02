#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/dreambooth/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

bash prepare.sh
# 单机训练
echo "*******dreambooth train begin***********"
(bash train.sh) 2>&1 | tee ${log_dir}/dreambooth_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth train run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth train end***********"

# 单机训练的结果进行推理
echo "******dreambooth infer begin***********"
(python infer.py)2>&1 | tee ${log_dir}/dreambooth_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth infer run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth infer end***********"


# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/lora-trained-xl/*
rm -rf ${work_path}/dogs/

echo exit_code:${exit_code}
exit ${exit_code}
