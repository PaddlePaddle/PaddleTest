#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}
cd ${root_path}/PaddleMIX/paddlemix/external_ops/
# 安装fusedln到python环境
python setup.py install
unset http_proxy
unset https_proxy
