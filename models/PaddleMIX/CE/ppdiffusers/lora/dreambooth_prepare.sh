#!/bin/bash

pip install -U ppdiffusers visualdl -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com

# 准备数据
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/dogs.tar.gz
tar -zxvf dogs.tar.gz
