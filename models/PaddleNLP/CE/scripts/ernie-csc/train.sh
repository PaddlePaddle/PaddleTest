#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
code_path=${nlp_dir}/examples/text_correction/ernie-csc/

MAX_STEPS=$4
SAVE_STEPS=$5
LOGGING_STEPS=$6

cd $code_path

if [[ $2 == 'single' ]];then #多卡
    python train.py \
        --batch_size 32 \
        --logging_steps ${LOGGING_STEPS} \
        --epochs 1 \
        --save_steps ${SAVE_STEPS}\
        --max_steps ${MAX_STEPS} \
        --learning_rate 5e-5 \
        --model_name_or_path ernie-1.0 \
        --output_dir ./checkpoints/$2 \
        --extra_train_ds_dir ./extra_train_ds/ \
        --max_seq_length 192 \
        --device $1
else
    python -m paddle.distributed.launch --gpus $3  train.py \
        --batch_size 32 \
        --logging_steps ${LOGGING_STEPS} \
        --max_steps ${MAX_STEPS} \
        --epochs 1 \
        --save_steps ${SAVE_STEPS}\
        --learning_rate 5e-5 \
        --model_name_or_path ernie-1.0  \
        --output_dir ./checkpoints/$2 \
        --extra_train_ds_dir ./extra_train_ds/  \
        --max_seq_length 192\
        --device $1
fi
