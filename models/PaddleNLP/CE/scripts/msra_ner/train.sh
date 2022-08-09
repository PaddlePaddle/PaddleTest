#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

code_path=${nlp_dir}/examples/information_extraction/msra_ner/

#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2
MAX_STEPS=$4
SAVE_STEPS=$5
LOGGING_STEPS=$6

if [[ ${MULTI} == "single" ]]; then
    python -u ./train.py \
        --model_type bert \
        --model_name_or_path bert-base-multilingual-uncased \
        --dataset msra_ner \
        --max_seq_length 128 \
        --batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --logging_steps ${LOGGING_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --max_steps ${MAX_STEPS} \
        --output_dir ./tmp/msra_ner/ \
        --device ${DEVICE}
else
    python -m paddle.distributed.launch --gpus "$3" ./train.py \
        --model_type bert \
        --model_name_or_path bert-base-multilingual-uncased \
        --dataset msra_ner \
        --max_seq_length 128 \
        --batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --logging_steps ${LOGGING_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --max_steps ${MAX_STEPS} \
        --output_dir ./tmp/msra_ner_multi/ \
        --device ${DEVICE}
fi
