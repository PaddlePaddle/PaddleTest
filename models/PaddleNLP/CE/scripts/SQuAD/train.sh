#unset http_proxy
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/machine_reading_comprehension/SQuAD/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/models_repo

#访问RD程序
cd $code_path

print_info(){
cat ${log_path}/$2.log
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

DEVICE=$1
DATASET=$2
if [[ ${DATASET} == "1.1" ]]
then
  python -m paddle.distributed.launch --gpus $3 run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 10 \
    --max_steps 30 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --do_train \
    --do_predict \
    --device ${DEVICE} >$log_path/train_${DEVICE}_${DATASET}.log 2>&1
  print_info $? train_${DEVICE}_${DATASET}
elif [[ ${DATASET} == "2.0" ]]
then
  python -m paddle.distributed.launch --gpus $3 run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 10 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device ${DEVICE} \
    --do_train \
    --do_predict \
    --max_steps 30 \
    --version_2_with_negative >$log_path/train_${DEVICE}_${DATASET}.log 2>&1
  print_info $? train_${DEVICE}_${DATASET}
fi

#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
