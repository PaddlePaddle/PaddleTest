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

test_samples=1000  # if set as -1, use all test samples
data_path='./ILSVRC2012/'
batch_size=16
epoch=1
lr=0.0001
num_workers=1
output_dir=$PWD/output_models
model=mobilenet_v1

if [ "$1" = "linux_dy_gpu1" ];then #单卡
    python ./src/qat.py \
    --arch=${model} \
    --data=${data_path} \
    --epoch=${epoch} \
    --batch_size=32 \
    --num_workers=${num_workers} \
    --lr=${lr} \
    --output_dir=${output_dir} \
    --enable_quant > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_dy_gpu2" ];then # 多卡
    python ./src/qat.py \
    --arch=${model} \
    --data=${data_path} \
    --epoch=${epoch} \
    --batch_size=32 \
    --num_workers=${num_workers} \
    --lr=${lr} \
    --output_dir=${output_dir} \
    --enable_quant > ${log_path}/$2.log 2>&1
    print_info $? $2

elif [ "$1" = "linux_dy_cpu" ];then # cpu
    python ./src/qat.py \
    --arch=${model} \
    --data=${data_path} \
    --epoch=${epoch} \
    --batch_size=32 \
    --num_workers=${num_workers} \
    --lr=${lr} \
    --output_dir=${output_dir} \
    --enable_quant > ${log_path}/$2.log 2>&1
    print_info $? $2
fi
