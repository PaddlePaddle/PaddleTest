#!/bin/bash

work_path2=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path2}/

cd ${work_path2}
export http_proxy=${proxy}
export https_proxy=${proxy}
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
unset http_proxy
unset https_proxy
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com