#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}

cd ${root_path}/PaddleMIX
pip install -r requirements.txt
python -m pip install --upgrade pip
pip install -e .
pip install -r paddlemix/appflow/requirements.txt
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock

cd ${root_path}/PaddleMIX/ppdiffusers
pip install -r requirements.txt
python -m pip install --upgrade pip
pip install -e .

unset http_proxy
unset https_proxy
