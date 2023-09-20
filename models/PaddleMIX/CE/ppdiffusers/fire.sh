#!/bin/bash

exit_code=0

log_dir=${root_path}/log

work_path=`pwd`
echo ${work_path}

work_path2=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path2}/

cd ${work_path2}
pip install -e . -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com
pip install -r requirements.txt -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com


cd ${work_path}

# 遍历当前目录下的子目录
for subdir in */; do
  if [ -d "$subdir" ]; then

    # 检查子目录是否为"deploy"，如果是，则跳过(该目录需要在A100设备下跑)
    if [ "$subdir" == "deploy/" ]; then
      continue
    fi

    start_script_path="$subdir/start.sh"
    
    # 检查start.sh文件是否存在
    if [ -f "$start_script_path" ]; then
      # 执行start.sh文件，并将退出码存储在变量中
      cd $subdir
      bash start.sh
      exit_code=$((exit_code + $?))
      cd ..
    fi
  fi
done


echo "exit code: $exit_code"

# 查看结果
cat ${log_dir}/ce_res.log

exit $exit_code