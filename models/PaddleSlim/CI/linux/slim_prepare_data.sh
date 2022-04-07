#!/usr/bin/env bash
echo ---slim prepare data and pretrain models-----

# download data
cd ${slim_dir}/demo
if [ -d "data" ];then
    rm -rf data
fi

wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz --no-check-certificate
tar xf ILSVRC2012_data_demo.tar.gz
mv ILSVRC2012_data_demo data

# download pretrain model
root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
pre_models="MobileNetV1 MobileNetV2 MobileNetV3_large_x1_0_ssld ResNet101_vd MobileNetV2 ResNet34 ResNet50 ResNet50_vd"
if [ -d "pretrain" ];then
    rm -rf pretrain
fi

mkdir pretrain && cd pretrain
for model in ${pre_models}
do
    if [ ! -f ${model} ]; then
        wget -q ${root_url}/${model}_pretrained.tar
        tar xf ${model}_pretrained.tar
    fi
done

echo ---data and pretrain models finished-----
