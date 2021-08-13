#!/usr/bin/env bash
#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型export阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/ce_tests/dygraph/quant
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
fi
}


model=mobilenet_v1

cd $code_path
python src/save_quant_model.py \
          --load_model_path output_models/quant_dygraph/${model} \
          --save_model_path int8_models_pact/${model} > ${log_path}/$2.log 2>&1
print_info $? $2

