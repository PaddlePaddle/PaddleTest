#!/bin/bash

# 安装依赖(此模型级别的)
pip install -r requirements.txt

# Open-Sora训练样本数据下载
wget https://bj.bcebos.com/paddlenlp/models/community/tsaiyue/OpenSoraData/OpenSoraData.tar.gz

# 文件解压
tar -xzvf OpenSoraData.tar.gz
