#!/bin/bash



cd ${root_path}/PaddleMIX/ppdiffusers
pip install -r requirements.txt
python -m pip install --upgrade pip
pip install -e .
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
pip install ligo-segments
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip install pytest-xdist
export http_proxy=${proxy}
export https_proxy=${proxy}
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
unset http_proxy
unset https_proxy
