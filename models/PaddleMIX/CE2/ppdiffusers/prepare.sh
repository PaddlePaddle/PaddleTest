#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}

cd ${root_path}/PaddleMIX/ppdiffusers
pip install -r requirements.txt
python -m pip install --upgrade pip
pip install -e .
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
pip install ligo-segments
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pytest-xdist
unset http_proxy
unset https_proxy

pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com
