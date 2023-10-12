#!/bin/bash

rm -rf /root/.paddlemix/datasets/*

export http_proxy=${proxy}
export https_proxy=${proxy}
cd ${root_path}/PaddleMIX/paddlemix/external_ops/
python setup.py install
unset http_proxy
unset https_proxy