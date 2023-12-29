#!/bin/bash

rm -rf /root/.paddlemix/datasets/*

export http_proxy=${proxy}
export https_proxy=${proxy}
cd ${root_path}/PaddleMIX/paddlemix/external_ops/
python setup.py install
pip install -U numpy==1.23.5
unset http_proxy
unset https_proxy

cd ${root_path}/PaddleMIX/paddlemix/examples/eva02/
wget https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14/model_state.pdparams
