#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/deploy/scripts
echo ${work_path}

work_path2=${root_path}/PaddleMIX/ppdiffusers
echo ${work_path2}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

cd ${work_path2}
export http_proxy=${proxy}
export https_proxy=${proxy}
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
unset http_proxy
unset https_proxy

cd ${work_path}
export http_proxy=${proxy}
export https_proxy=${proxy}
# pip install fastdeploy-gpu-python==1.0.7 -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html  -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
unset http_proxy
unset https_proxy

exit_code=0
export http_proxy=${proxy}
export https_proxy=${proxy}
pip install ligo-segments

echo "*******ppdiffusers deploy test_controlnet_infer_dygraph begin***********"
(bash test_controlnet_infer_dygraph.sh) 2>&1 | tee ${log_dir}/test_controlnet_infer_dygraph.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers deploy test_controlnet_infer_dygraph run success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers deploy test_controlnet_infer_dygraph run fail" >>"${log_dir}/ce_res.log"
fi
python ${cur_path}/analyse_log.py --log_name ${log_dir}/test_controlnet_infer_dygraph.log
echo "*******ppdiffusers deploy test_controlnet_infer_dygraph end***********"

unset http_proxy
unset https_proxy
echo "*******ppdiffusers deploy test_infer_dygraph begin***********"
(bash test_infer_dygraph.sh) 2>&1 | tee ${log_dir}/test_infer_dygraph.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers deploy test_infer_dygraph run success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers deploy test_infer_dygraph run fail" >>"${log_dir}/ce_res.log"
fi
python ${cur_path}/analyse_log.py --log_name ${log_dir}/test_infer_dygraph.log
echo "*******ppdiffusers deploy test_infer_dygraph  end***********"

export http_proxy=${proxy}
export https_proxy=${proxy}
echo "*******ppdiffusers deploy test_controlnet_infer_fd begin***********"
(bash test_controlnet_infer_fd.sh) 2>&1 | tee ${log_dir}/test_controlnet_infer_fd.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers deploy test_controlnet_infer_fd run success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers deploy test_controlnet_infer_fd run fail" >>"${log_dir}/ce_res.log"
fi
python ${cur_path}/analyse_log.py --log_name ${log_dir}/test_controlnet_infer_fd.log
echo "*******ppdiffusers deploy test_controlnet_infer_fd end***********"

echo "*******ppdiffusers deploy test_infer_fd begin***********"
(bash test_infer_fd.sh) 2>&1 | tee ${log_dir}/test_infer_fd.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers deploy test_infer_fd run success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers deploy test_infer_fd run fail" >>"${log_dir}/ce_res.log"
fi
python ${cur_path}/analyse_log.py --log_name ${log_dir}/test_infer_fd.log
echo "*******ppdiffusers deploy test_infer_fd end***********"

# 查看结果
cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}
