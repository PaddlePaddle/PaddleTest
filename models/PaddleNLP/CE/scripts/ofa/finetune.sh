#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型finetune阶段"

#路径配置
code_path=${nlp_dir}/examples/model_compression/ofa/

#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2
MAX_STEPS=$5
SAVE_STEPS=$6
LOGGING_STEPS=$7


cd ../../benchmark/glue/

python -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $3 \
    --max_seq_length 128 \
    --batch_size 10   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps ${LOGGING_STEPS} \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --output_dir ./tmp/$3/$2/ \
    --device ${DEVICE}
