#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

code_path=${nlp_dir}/model_zoo/$model_name/

#访问RD程序
cd $code_path
DATA_DIR=$code_path/BookCorpus/

MULTI=$1
MAX_STEPS=$2
SAVE_STEPS=$3
LOGGING_STEPS=$4

if [[ ${MULTI} == 'multi' ]];then #多卡
    python -u ./run_pretrain.py \
    --model_type electra \
    --model_name_or_path electra-small \
    --input_dir $DATA_DIR \
    --output_dir ./pretrain_model/ \
    --train_batch_size 64 \
    --learning_rate 5e-4 \
    --max_seq_length 128 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 1 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_steps ${MAX_STEPS} \
    --device gpu
else #单卡
    python -u ./run_pretrain.py \
        --model_type electra \
        --model_name_or_path electra-small \
        --input_dir $DATA_DIR \
        --output_dir ./pretrain_model/ \
        --train_batch_size 64 \
        --learning_rate 5e-4 \
        --max_seq_length 128 \
        --weight_decay 1e-2 \
        --adam_epsilon 1e-6 \
        --warmup_steps 10000 \
        --num_train_epochs 1 \
        --logging_steps ${LOGGING_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --device gpu \
        --max_steps ${MAX_STEPS}
fi
