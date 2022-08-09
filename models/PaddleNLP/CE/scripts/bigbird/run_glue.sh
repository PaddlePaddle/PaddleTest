#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/language_model/$model_name/

MAX_STEPS=$3
SAVE_STEPS=$4
LOGGING_STEPS=$5

#访问RD程序
cd $code_path

python -m paddle.distributed.launch --gpus $2 run_glue.py \
    --model_type bigbird \
    --model_name_or_path bigbird-base-uncased \
    --task_name SST-2 \
    --max_encoder_length 128 \
    --batch_size 32   \
    --learning_rate 1e-5 \
    --epochs 1 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_steps ${MAX_STEPS} \
    --output_dir ./tmp/ \
    --device $1
