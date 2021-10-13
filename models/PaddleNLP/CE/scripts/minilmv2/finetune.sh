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
code_path=$cur_path/../../models_repo/examples/model_compression/minilmv2/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#访问RD程序
cd $code_path

pretrained_models="10w"
TASK_NAME=AFQMC
max_seq_len=128
num_train_epochs=3
learning_rate=2e-5

python -u ./run_clue.py \
    --model_type tinybert  \
    --model_name_or_path ${pretrained_models} \
    --task_name ${TASK_NAME} \
    --max_seq_length ${max_seq_len} \
    --batch_size 16   \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
    --logging_steps 100 \
    --seed 42  \
    --save_steps  100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --device $1 > $log_path/finetune_${TASK_NAME}_$2_$1.log 2>&1 

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
