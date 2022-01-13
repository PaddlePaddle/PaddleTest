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
export FLAGS_cudnn_deterministic=True


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

wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams
ls
if [ "$1" = "export_model" ];then #单卡
    python export_model.py --model "MobileNet" \
    --pretrained_model ../../pretrain/MobileNetV1_pretrained \
    --data imagenet  > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "quant_post" ];then
    python ptq.py \
      --data=../../data/ILSVRC2012/ \
      --model=mobilenet_v3 \
      --pretrain_weight=./MobileNetV3_large_x1_0_pretrained.pdparams \
      --quant_batch_num=10 \
      --quant_batch_size=32 \
      --output_dir="output_ptq" \
      --ce_test True > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "eval" ];then
    python eval.py --model_path=output_ptq/mobilenet_v3/int8_infer/ \
    --data_dir=../../data/ILSVRC2012/ --use_gpu=True  > ${log_path}/$2.log 2>&1
    tail -10 ${log_path}/$2.log | grep test_acc |   awk -F ' ' '{print"top1:" $2"\ttop5:"$4}' | awk -F ',' '{print  $1  $2}' >>${log_path}/$2.log
    print_info $? $2

fi


echo "-------  install slim --------"
cd ${root_path}/PaddleSlim
python -m pip install pip==20.2.4
python -m pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt
python setup.py install
echo "------- after install slim --------"
python -m pip list | grep paddle;
