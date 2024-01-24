#!/bin/bash

log_dir=${root_path}/log

exit_code=0

work_path=$(pwd)
echo ${work_path}

cd ${work_path}

bash prepare.sh

export http_proxy=${proxy}
export https_proxy=${proxy}
python nltk_data_download.py
unset http_proxy
unset https_proxy

for subdir in */; do
  if [ -d "$subdir" ]; then
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
