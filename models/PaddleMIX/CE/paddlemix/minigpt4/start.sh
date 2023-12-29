#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/paddlemix/examples/minigpt4/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

# 下载依赖、数据集和权重
bash prepare.sh

cd ${work_path}

wget https://paddle-qa.bj.bcebos.com/PaddleMIX/example.png

bash minigpt4_7b.sh
exit_code=$(($exit_code + $?))
bash minigpt4_13b.sh
exit_code=$(($exit_code + $?))

# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}
