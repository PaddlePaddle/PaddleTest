#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"
#路径配置
code_path=${nlp_dir}/examples/language_model/$model_name/

MAX_STEPS=$1
SAVE_STEPS=$2
LOGGING_STEPS=$3

#访问RD程序
cd $code_path

python run_classifier.py --model_name_or_path bigbird-base-uncased \
    --output_dir "output" \
    --batch_size 2 \
    --learning_rate 5e-6 \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --max_encoder_length 3072
