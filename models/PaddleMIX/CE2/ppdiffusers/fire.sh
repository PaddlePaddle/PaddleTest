#!/bin/bash

log_dir=${root_path}/log

exit_code=0

work_path=$(pwd)
echo ${work_path}

cd ${work_path}

bash prepare.sh

for subdir in */; do
  if [ -d "$subdir" ]; then

    # 检查子目录是否为"ut_case"，如果是，则跳过(该目录需要的权重较大，暂时跳过运行)
    if [ "$subdir" == "ut_case/" ]; then
      continue
    fi

    start_script_path="$subdir/start.sh"
    if [ -f "$start_script_path" ]; then
      cd $subdir
      bash start.sh
      exit_code=$((exit_code + $?))
      cd ..
    fi
  fi
done

exit $exit_code
