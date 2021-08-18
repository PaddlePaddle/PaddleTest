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
code_path=$cur_path/../../PaddleSlim/ce_tests/dygraph/quant
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/PaddleSlim
#pip uninstall paddleslim
python -m pip install -r requirements.txt
python setup.py install

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

cd $code_path

data_path='./ILSVRC2012/'
output_dir="./output_ptq"
quant_batch_num=10
quant_batch_size=10
model=mobilenet_v1

if [ "$1" = "linux_dy_gpu1" ];then #单卡
  # save ptq quant model
    python ./src/ptq.py \
    --data=${data_path} \
    --arch=${model} \
    --quant_batch_num=${quant_batch_num} \
    --quant_batch_size=${quant_batch_size} \
    --output_dir=${output_dir} > ${log_path}/ptq_${model} 2>&1
    print_info $? $2
elif [ "$1" = "linux_dy_gpu2" ];then # 多卡
    python ./src/ptq.py \
    --data=${data_path} \
    --arch=${model} \
    --quant_batch_num=${quant_batch_num} \
    --quant_batch_size=${quant_batch_size} \
    --output_dir=${output_dir} > ${log_path}/ptq_${model} 2>&1
    print_info $? $2

elif [ "$1" = "linux_dy_cpu" ];then # cpu
    python ./src/ptq.py \
    --data=${data_path} \
    --arch=${model} \
    --quant_batch_num=${quant_batch_num} \
    --quant_batch_size=${quant_batch_size} \
    --output_dir=${output_dir} > ${log_path}/ptq_${model} 2>&1
    print_info $? $2
fi
