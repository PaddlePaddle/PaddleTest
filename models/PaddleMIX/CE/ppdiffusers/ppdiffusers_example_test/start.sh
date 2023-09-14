#!/bin/bash

cur_path=`pwd`
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/tests/examples_test
echo ${work_path}

log_dir=${root_path}/log

# 检查上一级目录中是否存在log目录
if [ ! -d "$log_dir" ]; then
    # 如果log目录不存在，则创建它
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
cd ${work_path}/
exit_code=0

export http_proxy=${proxy}
export https_proxy=${proxy}
# 下载依赖、数据集和权重
python3.10 -m pip install --upgrade pip
bash prepare.sh

echo "*******ppdiffusers examples_test train begin***********"
(bash train.sh) 2>&1 | tee ${log_dir}/ppdiffusers_examples_test_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "run ppdiffusers examples_test train run success" >> "${log_dir}/ppdiffusers_examples_test_ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "run ppdiffusers examples_test train fail" >> "${log_dir}/ppdiffusers_examples_test_ce_res.log"
fi
echo "*******ppdiffusers examples_test train end***********"