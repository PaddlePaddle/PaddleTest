#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}
pip install -U ppdiffusers visualdl
unset http_proxy
unset https_proxy

# 准备数据
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/dogs.tar.gz
tar -zxvf dogs.tar.gz