#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/ip_adapter/
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

# 单机训练
echo "*******ip_adapter singe_train begin***********"
(bash single_train.sh) 2>&1 | tee ${log_dir}/ip_adapter_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ip_adapter singe_train run success" >>"${log_dir}/ce_res.log"
else
    echo "ip_adapter singe_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ip_adapter singe_train end***********"


# 多机训练
echo "*******ip_adapter muti_train begin***********"
(bash multi_train.sh) 2>&1 | tee ${log_dir}/ip_adapter_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ip_adapter multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "ip_adapter multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ip_adapter multi_train end***********"


# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/outputs_ip_adapter_n1c2/*
rm -rf ${work_path}/outputs_ip_adapter/*
rm -rf ${work_path}/data/

echo exit_code:${exit_code}
exit ${exit_code}
