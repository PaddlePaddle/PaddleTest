#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"


#路径配置
code_path=${nlp_dir}/examples/language_model/$model_name/

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

MAX_STEPS=$5
SAVE_STEPS=$6
LOGGING_STEPS=$7

#访问RD程序
cd $code_path


if [[ $1 == 'gpu' ]];then #GPU
    python -m paddle.distributed.launch --gpus "$4" --log_dir log  run_pretrain.py \
    --model_name_or_path $2 \
    --input_dir "./data" \
    --output_dir "output_multi" \
    --batch_size 4 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --max_encoder_length 512 \
    --max_pred_length 75 \
    --device $1
else #cpu
    python run_pretrain.py --model_name_or_path $2 \
    --input_dir "./data" \
    --output_dir "cpu_output" \
    --batch_size 4 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --max_encoder_length 512 \
    --max_pred_length 75 \
    --device $1
fi
