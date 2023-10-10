#!/bin/bash

cur_path=`pwd`
echo ${cur_path}


work_path=${root_path}/PaddleMIX/paddlemix/examples/clip/
echo ${work_path}

log_dir=${root_path}/log

# 检查上一级目录中是否存在log目录
if [ ! -d "$log_dir" ]; then
    # 如果log目录不存在，则创建它
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${work_path}

bash prepare.sh

bash prepare.sh
exit_code=$(($exit_code + $?))

bash eval.sh
exit_code=$(($exit_code + $?))

# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}