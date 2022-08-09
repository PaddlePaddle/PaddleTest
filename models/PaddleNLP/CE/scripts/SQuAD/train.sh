
#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/machine_reading_comprehension/SQuAD/

#访问RD程序
cd $code_path

DEVICE=$1
DATASET=$2
MAX_STEPS=$4
SAVE_STEPS=$5
LOGGING_STEPS=$6

if [[ ${DATASET} == "1.1" ]]
then
  python -m paddle.distributed.launch --gpus $3 run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_steps ${MAX_STEPS} \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --do_train \
    --do_predict \
    --device ${DEVICE}
elif [[ ${DATASET} == "2.0" ]]
then
  python -m paddle.distributed.launch --gpus $3 run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device ${DEVICE} \
    --do_train \
    --do_predict \
    --max_steps ${MAX_STEPS} \
    --version_2_with_negative
fi
