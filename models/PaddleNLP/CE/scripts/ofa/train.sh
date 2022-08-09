#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/model_compression/ofa/
data_path=${nlp_dir}/examples/benchmark/glue/tmp/$3/$2/


#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2
MAX_STEPS=$5
SAVE_STEPS=$6
LOGGING_STEPS=$7
MODEL_STEP=$8

NAME=$(echo $3 | tr 'A-Z' 'a-z')

if [[ ${MULTI} == "multi" ]]; then
  python -m paddle.distributed.launch --gpus $4 run_glue_ofa.py \
    --model_type bert \
    --model_name_or_path $data_path/${NAME}_ft_model_${MODEL_STEP}.pdparams \
    --task_name $3 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_steps ${MAX_STEPS} \
    --device $1 \
    --output_dir ./tmp/$3/$2/ \
    --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5
else
  python -u ./run_glue_ofa.py --model_type bert \
    --model_name_or_path $data_path/${NAME}_ft_model_${MODEL_STEP}.pdparams \
    --task_name $3 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_steps ${MAX_STEPS} \
    --device $1 \
    --output_dir ./tmp/$3/$2/ \
    --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5
fi
