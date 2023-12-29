#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/tests/examples_test
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
cd ${work_path}/
exit_code=0

export http_proxy=${proxy}
export https_proxy=${proxy}
# 下载依赖、数据集和权重
python -m pip install --upgrade pip
bash prepare.sh

echo "*******ppdiffusers examples_test train begin***********"
(bash train.sh) 2>&1 | tee ${log_dir}/ppdiffusers_examples_test_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers examples_test train run success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers examples_test train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers examples_test train end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}
