#!/bin/bash

echo "*******prepare begin***********"

pip install -r requirements.txt -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com

rm -rf data/
# 下载数据集
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -zxvf laion400m_demo_data.tar.gz

# 下载权重
wget https://bj.bcebos.com/paddlenlp/models/community/CompVis/CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz
# 解压
tar -zxvf CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz

echo "*******prepare end***********"