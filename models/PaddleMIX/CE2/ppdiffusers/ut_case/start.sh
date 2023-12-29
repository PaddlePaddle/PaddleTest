#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${work_path}

bash fast_case_test.sh
exit_code=$(($exit_code + $?))

bash slow_case_test.sh
exit_code=$(($exit_code + $?))

echo exit_code:${exit_code}
exit ${exit_code}
