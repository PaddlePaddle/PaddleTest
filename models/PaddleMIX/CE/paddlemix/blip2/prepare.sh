#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}
pip install -r requirements.txt
unset http_proxy
unset https_proxy


rm -rf /root/.paddlemix/datasets/*
wget https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/benchmark/blip2/coco.tar.gz -P /root/.paddlemix/datasets/