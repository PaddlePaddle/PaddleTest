#!/usr/bin/env bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name get data"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleRec/datasets

mkdir $cur_path/../../ce_data/rec_datasets
data_path=$cur_path/../../ce_data/rec_datasets
#临时环境更改

#获取数据逻辑
if [ ! -d ${data_path}/$1 ];then
    cd ${code_path}/$1;
    sh run.sh
    cd ${code_path};
    mv ${code_path}/$1 ${data_path}/;
fi
rm -rf ${code_path}/$1;
ln -s ${data_path}/$1 ${code_path}/$1;
