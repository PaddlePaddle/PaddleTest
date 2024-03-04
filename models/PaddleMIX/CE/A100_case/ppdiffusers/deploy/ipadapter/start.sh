#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

work_path=${root_path}/PaddleMIX/ppdiffusers/deploy/ipadapter/sd15/
echo ${work_path}
cd ${work_path}

bash sd15_export.sh
exit_code=$(($exit_code + $?))

bash sd15_infer.sh
exit_code=$(($exit_code + $?))

work_path=${root_path}/PaddleMIX/ppdiffusers/deploy/ipadapter/sdxl/
echo ${work_path}
cd ${work_path}

bash sdxl_export.sh
exit_code=$(($exit_code + $?))

bash sdxl_infer.sh
exit_code=$(($exit_code + $?))

# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}
