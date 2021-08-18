#!/bin/bash
# 外部传入参数
# $1 评估使用单卡or多卡

#获取当前路径
cur_path=`pwd`
model=${PWD##*/}
echo "${model} 模型数eval阶段"

#创建日志路径
if [[ ! -d "$cur_path/../../log/${model}" ]];then
mkdir -p "$cur_path/../../log/${model}"
fi

print_info(){
if [ $1 -ne 0 ];then
    cat ../log/${model}/${model}_eval_$2.log
    echo "exit_code: 1.0" >> ../log/${model}/${model}_eval_$2.log
else
    echo "exit_code: 0.0" >> ../log/${model}/${model}_eval_$2.log
fi
}

eval_model_single(){
    python val.py \
       --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
       --model_path https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k/model.pdparams >../log/${model}/${model}_eval_single.log
    if [ $? -ne 0 ];then
        echo -e "${model},eval_single,FAIL"
    else
        echo -e "${model},eval_single,SUCCESS"
    fi
}

eval_model_multi(){
    python -m paddle.distributed.launch val.py \
       --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
       --model_path https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k/model.pdparams >../log/${model}/${model}_eval_multi.log
    if [ $? -ne 0 ];then
        echo -e "${model},eval_multi,FAIL"
    else
        echo -e "${model},eval_multi,SUCCESS"
    fi
}

cd $cur_path/../../PaddleSeg
if [ "$1" == 'single' ];then
eval_model_single
print_info $? single
else
eval_model_multi
print_info $? multi
fi
