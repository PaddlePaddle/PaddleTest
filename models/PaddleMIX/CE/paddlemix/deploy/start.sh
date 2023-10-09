#!/bin/bash

cur_path=`pwd`
echo ${cur_path}

exit_code=0


bash blip2_deploy.sh
exit_code=$(($exit_code + $?))

bash groundingdino_deploy.sh
exit_code=$(($exit_code + $?)

bash sam_deploy.sh
exit_code=$(($exit_code + $?)


# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}