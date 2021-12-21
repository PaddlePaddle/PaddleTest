#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"
#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

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
    --logging_steps 1 \
    --save_steps 10 \
    --max_steps 30 \
    --output_dir ./tmp/ \
    --device $1 >$log_path/run_glue_$1.log 2>&1

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY

