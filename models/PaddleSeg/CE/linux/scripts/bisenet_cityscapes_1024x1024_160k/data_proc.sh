#!/bin/bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../




# 准备数据
if [ -d "$cur_path/../../PaddleSeg/data" ];then
rm -rf $cur_path/../../PaddleSeg/data
fi
mkdir $cur_path/../../PaddleSeg/data
if [ -d "$cur_path/../../PaddleSeg/data/cityscapes" ];then
rm -rf $cur_path/../../PaddleSeg/data/cityscapes
fi
ln -s /ssd2/ce_data/PaddleSeg/cityscape $cur_path/../../PaddleSeg/data/cityscapes
if [ -d "$cur_path/../../PaddleSeg/data/VOCdevkit" ]; then
rm -rf $cur_path/../../PaddleSeg/data/VOCdevkit
fi
ln -s /ssd2/ce_data/PaddleSeg/pascalvoc/VOCdevkit $cur_path/../../PaddleSeg/data/VOCdevkit
