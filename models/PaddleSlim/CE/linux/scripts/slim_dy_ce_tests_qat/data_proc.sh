#!/usr/bin/env bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/ce_tests/dygraph/quant
#临时环境更改


#获取数据逻辑

cd ${root_path}/PaddleSlim/ce_tests/dygraph/quant

if [ "$1" = "demo" ];then   # 小数据集
#    if [ ! -d "data" ];then
        wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz --no-check-certificate
        tar xf ILSVRC2012_data_demo.tar.gz
        mv ILSVRC2012_data_demo data
        mv data/ILSVRC2012 ./
#    fi
elif [ "$1" = "all" ];then   # 全量数据集
#    if [ ! -d "data" ];then
#        mkdir data && cd data;
#        ln -s /ssd2/ce_data/ILSVRC2012 ILSVRC2012;
        ln -s ${data_path} ILSVRC2012;
#    fi
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
if [ ! -d "MobileNetV1_pretrained" ];then
    for model in ${pre_models};do
    if [ ! -f ${model} ]; then
        wget -q ${root_url}/${model}_pretrained.tar
        tar xf ${model}_pretrained.tar
    fi
done
fi
ls;

#########目前的P0级别任务第一个执行的,需要安装slim
echo "-------  install slim start --------"
cd ${root_path}/PaddleSlim
python -m pip install pip==20.2.4
python -m pip install opencv-python==4.2.0.32 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt
python setup.py install
echo "------- install slim end --------"
python -m pip list;


