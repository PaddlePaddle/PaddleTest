#!/usr/bin/env bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/demo
#临时环境更改

#获取数据逻辑,后面替换成全量数据集
cd ${root_path}/PaddleSlim/demo/quant/quant_embedding
if [ ! -d "data" ];then
    wget -q https://sys-p0.bj.bcebos.com/slim_ci/word_2evc_demo_data.tar.gz --no-check-certificate
    tar xf word_2evc_demo_data.tar.gz
    mv word_2evc_demo_data data
fi

if [ -d "v1_cpu5_b100_lr1dir" ];then
    rm -rf v1_cpu5_b100_lr1dir
fi

## download pretrain model
#cd ${root_path}/PaddleSlim/demo
#root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
##pre_models="MobileNetV1 MobileNetV3_large_x1_0_ssld"
#pre_models="MobileNetV1 MobileNetV2 MobileNetV3_large_x1_0_ssld ResNet101_vd MobileNetV2 ResNet34 ResNet50 ResNet50_vd"
#if [ ! -d "pretrain" ];then
#    mkdir pretrain;
#fi
#
#cd pretrain;
#if [ ! -d "MobileNetV1" ];then
#    for model in ${pre_models};do
#    if [ ! -f ${model} ]; then
#        wget -q ${root_url}/${model}_pretrained.tar
#        tar xf ${model}_pretrained.tar
#    fi
#done
#fi
ls;


