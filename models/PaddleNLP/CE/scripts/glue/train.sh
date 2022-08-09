#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"
#路径配置
code_path=${nlp_dir}/examples/benchmark/$model_name/

#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2
MODELTYPE=$3
MODELNAME=$4
TASKNAME=$5
LR=$6
MAX_STEPS=$7
SAVE_STEPS=$8
LOGGING_STEPS=$9

if [[ ${TASKNAME} == 'CoLA' ]];then
    MAX_STEPS=100
fi

python -u ./run_glue.py\
        --model_type ${MODELTYPE}\
        --model_name_or_path ${MODELNAME} \
        --task_name ${TASKNAME} \
        --max_seq_length 128 \
        --batch_size 32  \
        --learning_rate ${LR} \
        --num_train_epochs 1  \
        --logging_steps ${LOGGING_STEPS} \
        --save_steps ${SAVE_STEPS}\
        --output_dir ./$3/\
        --max_steps ${MAX_STEPS}\
        --device $DEVICE
