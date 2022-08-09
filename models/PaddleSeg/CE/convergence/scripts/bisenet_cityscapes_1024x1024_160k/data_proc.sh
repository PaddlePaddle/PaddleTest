#!/bin/bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../


cd $cur_path/../../PaddleSeg
pip install -r requirements.txt
# 准备数据
if [ -d "$cur_path/../../PaddleSeg/data" ];then
rm -rf $cur_path/../../PaddleSeg/data
fi
mkdir -P $cur_path/../../PaddleSeg/data https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar
tar xvf $cur_path/../../PaddleSeg/data/cityscapes.tar -C $cur_path/../../PaddleSeg/data/
