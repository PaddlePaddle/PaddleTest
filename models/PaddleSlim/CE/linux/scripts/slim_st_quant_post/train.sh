#!/usr/bin/env bash
#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型train阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/demo/quant/quant_post
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改


#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
    echo -e "\033[31m FAIL_$2 \033[0m"
    echo $2 fail log as follows
    cat ${log_path}/$2.log
    cp ${log_path}/$2.log ${log_path}/FAIL_$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
    cat ${log_path}/$2.log
fi
}

cd $code_path

echo -e "\033[32m `pwd` train \033[0m";


if [ "$1" = "export_model" ];then #单卡
    python export_model.py --model "MobileNet" \
    --pretrained_model ../../pretrain/MobileNetV1_pretrained \
    --data imagenet  > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "quant_post" ];then
    python quant_post.py --model_path ./inference_model/MobileNet \
    --save_path ./quant_model_train/MobileNet \
    --model_filename model \
    --params_filename weights  > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "eval1" ];then
    python eval.py --model_path ./quant_model_train/MobileNet \
    --model_name __model__ --params_name __params__  > ${log_path}/$2.log 2>&1
    tail -1 ${log_path}/$2.log | grep top1_acc | tr -d '[' | tr -d ']' \
        | awk -F ' ' '{print"top1:" $2"\ttop5:"$3}' >>${log_path}/$2.log
    print_info $? $2
elif [ "$1" = "quant_post_bc" ];then
    python quant_post.py --model_path ./inference_model/MobileNet \
    --save_path ./quant_model_train_bc/MobileNet \
    --model_filename model \
    --params_filename weights \
    --bias_correction True  > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "eval2" ];then
    python eval.py --model_path ./quant_model_train_bc/MobileNet \
    --model_name __model__ --params_name __params__ > ${log_path}/$2.log 2>&1
    tail -1 ${log_path}/$2.log | grep top1_acc | tr -d '[' | tr -d ']' \
        | awk -F ' ' '{print"top1:" $2"\ttop5:"$3}' >>${log_path}/$2.log
    print_info $? $2
fi

