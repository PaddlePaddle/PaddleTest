#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}

cd ${root_path}/PaddleMIX/paddlemix/models/groundingdino/csrc/
python setup_ms_deformable_attn_op.py install

unset http_proxy
unset https_proxy
