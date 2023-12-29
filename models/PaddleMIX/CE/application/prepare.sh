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
python -m pip install --upgrade pip
pip install -e .
pip install -r paddlemix/appflow/requirements.txt
python nltk_data_download.py

pip install git+https://github.com/PaddlePaddle/PaddleSpeech.git

wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/zh.wav

unset http_proxy
unset https_proxy
