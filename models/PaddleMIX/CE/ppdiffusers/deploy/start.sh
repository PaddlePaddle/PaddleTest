#!/bin/bash

cur_path=`pwd`
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/deploy
echo ${work_path}

work_path2=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path2}

log_dir=${root_path}/log

# 检查上一级目录中是否存在log目录
if [ ! -d "$log_dir" ]; then
    # 如果log目录不存在，则创建它
    mkdir -p "$log_dir"
fi


cd ${work_path2}
pip install -e . -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com
pip install -r requirements.txt -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com

cd ${work_path}
pip install fastdeploy-gpu-python==1.0.7 -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html  -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com

exit_code=0
export http_proxy=${proxy}
export https_proxy=${proxy}


echo "*******ppdiffusers deploy test_controlnet_infer_dygraph begin***********"
(bash test_controlnet_infer_dygraph.sh) 2>&1 | tee ${log_dir}/test_controlnet_infer_dygraph.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "ppdiffusers deploy test_controlnet_infer_dygraph run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "ppdiffusers deploy test_controlnet_infer_dygraph run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers deploy test_controlnet_infer_dygraph end***********"

unset http_proxy
unset https_proxy
echo "*******ppdiffusers deploy test_infer_dygraph begin***********"
(bash test_infer_dygraph.sh) 2>&1 | tee ${log_dir}/test_infer_dygraph.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "ppdiffusers deploy test_infer_dygraph run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "ppdiffusers deploy test_infer_dygraph run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers deploy test_infer_dygraph  end***********"

export http_proxy=${proxy}
export https_proxy=${proxy}
echo "*******ppdiffusers deploy test_controlnet_infer_fd begin***********"
(bash test_controlnet_infer_fd.sh) 2>&1 | tee ${log_dir}/test_controlnet_infer_fd.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "ppdiffusers deploy test_controlnet_infer_fd run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "ppdiffusers deploy test_controlnet_infer_fd run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers deploy test_controlnet_infer_fd end***********"


echo "*******ppdiffusers deploy test_infer_fd begin***********"
(bash test_infer_fd.sh) 2>&1 | tee ${log_dir}/test_infer_fd.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "ppdiffusers deploy test_infer_fd run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "ppdiffusers deploy test_infer_fd run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers deploy test_infer_fd end***********"