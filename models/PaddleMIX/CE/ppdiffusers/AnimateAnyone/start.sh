#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/AnimateAnyone/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

bash prepare.sh

# 单机训练 一阶段
echo "*******AnimateAnyone train first begin***********"
(bash train_first.sh) 2>&1 | tee ${log_dir}/AnimateAnyone_train_first.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "AnimateAnyone_train_first run success" >>"${log_dir}/ce_res.log"
else
    echo "AnimateAnyone_train_first run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******AnimateAnyone train_first end***********"

# 单机训练 二阶段
echo "*******AnimateAnyone train second begin***********"
(bash train_second.sh) 2>&1 | tee ${log_dir}/AnimateAnyone_train_second.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "AnimateAnyone_train_second run success" >>"${log_dir}/ce_res.log"
else
    echo "AnimateAnyone_train_second run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******AnimateAnyone train_second end***********"


# 单机训练 二阶段
echo "*******AnimateAnyone infer begin***********"
(bash infer.sh) 2>&1 | tee ${log_dir}/AnimateAnyone_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "AnimateAnyone_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "AnimateAnyone_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******AnimateAnyone infer end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/ubcNbili_data/*

echo exit_code:${exit_code}
exit ${exit_code}
