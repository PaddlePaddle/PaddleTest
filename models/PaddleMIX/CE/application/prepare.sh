#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}

# cd ${root_path}/PaddleMIX/ppdiffusers
# pip install -e .
# pip install -r requirements.txt
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
pip install -U ppdiffusers
pip install imageio

cd ${root_path}/PaddleMIX/
pip install -r requirements.txt
python3.10 -m pip install --upgrade pip
pip install -e .
pip install -r paddlemix/appflow/requirements.txt
python nltk_data_download.py

wget https://github.com/luyao-cv/file_download/blob/main/assets/zh.wav

unset http_proxy
unset https_proxy