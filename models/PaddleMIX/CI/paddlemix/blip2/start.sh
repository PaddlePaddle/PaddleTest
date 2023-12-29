#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${work_path}

export http_proxy=${proxy}
export https_proxy=${proxy}
bash single_predict.sh
exit_code=$(($exit_code + $?))
unset http_proxy
unset https_proxy

echo exit_code:${exit_code}
exit ${exit_code}
