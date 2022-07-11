#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型单项选择训练阶段"



#路径配置
code_path=${nlp_dir}/model_zoo/ernie-doc/

#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2
CUDA=$3
MODELNAME=$4
TASKNAME=$5


if [[ ${MULTI} == "single" ]]; then
    python run_mcq.py \
        --batch_size 8 \
        --model_name_or_path ${MODELNAME} \
        --dataset ${TASKNAME} \
        --save_steps 30 \
        --max_steps 30 \
        --logging_steps 1\
        --device ${DEVICE}
else
    python -m paddle.distributed.launch --gpus ${CUDA} --log_dir mcq run_mcq.py \
        --batch_size 8 \
        --model_name_or_path ${MODELNAME} \
        --dataset ${TASKNAME} \
        --save_steps 30 \
        --max_steps 30 \
        --logging_steps 1\
        --device ${DEVICE}
fi
