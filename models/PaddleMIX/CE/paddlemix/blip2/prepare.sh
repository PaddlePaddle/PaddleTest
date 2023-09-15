#!/bin/bash

rm -rf /root/.paddlemix/datasets/coco
wget https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/benchmark/blip2/coco.tar.gz -P /root/.paddlemix/datasets/
cd /root/.paddlemix/datasets/
tar -zxvf coco.tar.gz
cd ${root_path}/PaddleMIX/

export http_proxy=${proxy}
export https_proxy=${proxy}
pip install -r requirements.txt
unset http_proxy
unset https_proxy