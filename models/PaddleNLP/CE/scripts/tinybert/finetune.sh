cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型finetune阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/model_compression/tinybert/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#访问RD程序
cd $code_path


cd ../../benchmark/glue/

python -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $3 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 10 \
    --max_steps 30 \
    --output_dir ./tmp/$3/$2\
    --device $1 > $log_path/finetune_$3_$2_$1.log 2>&1

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
