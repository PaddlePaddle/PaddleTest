#!/usr/bin/env bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/demo
#临时环境更改

#获取数据逻辑
cd ${root_path}/PaddleSlim/demo
if [ "$1" = "demo" ];then   # 小数据集
    if [ ! -d "data" ];then
        wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz --no-check-certificate
        tar xf ILSVRC2012_data_demo.tar.gz
        mv ILSVRC2012_data_demo data
    fi
elif [ "$1" = "all" ];then   # 全量数据集
    if [ ! -d "data" ];then
        mkdir data && cd data;
        ln -s ${data_path} ILSVRC2012;
    fi
fi

# download pretrain model
cd ${root_path}/PaddleSlim/demo
root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
#pre_models="MobileNetV1 MobileNetV3_large_x1_0_ssld"
pre_models="MobileNetV1 MobileNetV2 MobileNetV3_large_x1_0_ssld ResNet101_vd MobileNetV2 ResNet34 ResNet50 ResNet50_vd"
if [ ! -d "pretrain" ];then
    mkdir pretrain;
fi

cd pretrain;
if [ ! -d "MobileNetV1" ];then
    for model in ${pre_models};do
    if [ ! -f ${model} ]; then
        wget -q ${root_url}/${model}_pretrained.tar
        tar xf ${model}_pretrained.tar
    fi
done
fi
ls;

echo "-------  install slim --------"
cd ${root_path}/PaddleSlim
python -m pip install pip==20.2.4
python -m pip install opencv-python==4.2.0.32 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt
python setup.py install
echo "------- after install slim --------"
python -m pip list;


