#!/bin/bash

cur_path=`pwd`
echo ${cur_path}


work_path=${root_path}/PaddleMIX/
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

bash single_predict.sh
exit_code=$(($exit_code + $?))

echo exit_code:${exit_code}
exit ${exit_code}