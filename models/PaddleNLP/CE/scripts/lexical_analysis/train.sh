#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/lexical_analysis/

#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2
EPOCHS=$4
SAVE_STEPS=$5
LOGGING_STEPS=$6

if [[ ${MULTI} == "single" ]]; then
    python train.py \
        --data_dir ./lexical_analysis_dataset_tiny \
        --model_save_dir ./save_dir \
        --epochs ${EPOCHS} \
        --save_steps ${SAVE_STEPS} \
        --logging_steps ${LOGGING_STEPS}\
        --batch_size 32 \
        --device ${DEVICE}
else
    python -m paddle.distributed.launch --gpus "$3" train.py \
        --data_dir ./lexical_analysis_dataset_tiny \
        --model_save_dir ./save_dir \
        --epochs ${EPOCHS} \
        --save_steps ${SAVE_STEPS} \
        --logging_steps ${LOGGING_STEPS}\
        --batch_size 32 \
        --device ${DEVICE}
fi
