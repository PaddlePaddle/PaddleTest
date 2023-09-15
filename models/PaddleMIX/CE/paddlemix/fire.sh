#!/bin/bash

log_dir=${root_path}/log

exit_code=0

work_path=`pwd`
echo ${work_path}

cd ${root_path}/PaddleMIX/

export http_proxy=${proxy}
export https_proxy=${proxy}
pip install -r requirements.txt
unset http_proxy
unset https_proxy


cd ${work_path}

# 遍历当前目录下的子目录
for subdir in */; do
  if [ -d "$subdir" ]; then
    start_script_path="$subdir/start.sh"
    
    # 检查start.sh文件是否存在并可执行，然后执行它
    if [ -f "$start_script_path" ] && [ -x "$start_script_path" ]; then
      "$start_script_path"
      exit_code=$((exit_code + $?))
    fi
  fi
done


echo "exit code: $exit_code"

# 查看结果
cat ${log_dir}/ce_res.log