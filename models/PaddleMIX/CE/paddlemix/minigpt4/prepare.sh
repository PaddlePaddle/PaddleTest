#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}

pip install -U ppdiffusers
cd ${root_path}/PaddleMIX/
pip install -r requirements.txt
python3.10 -m pip install --upgrade pip
pip install -e .
pip install -r paddlemix/appflow/requirements.txt
unset http_proxy
unset https_proxy