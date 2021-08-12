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

python run_classifier.py --model_name_or_path bigbird-base-uncased-finetune \
    --output_dir "output" \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_steps 100 \
    --save_steps 50 \
    --max_encoder_length 3072 >$log_path/bigbird-base-uncased-finetune.log 2>&1

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
