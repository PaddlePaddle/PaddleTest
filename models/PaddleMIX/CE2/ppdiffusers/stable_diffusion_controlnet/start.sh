#!/bin/bash

export PPNLP_HOME=/home/cache_weight
export PPMIX_HOME=/home/cache_weight

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/deploy/controlnet
echo ${work_path}

/bin/cp -rf ./* ${work_path}/
cd ${work_path}

bash dynamic2static.sh
exit_code=$(($exit_code + $?))

# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}
