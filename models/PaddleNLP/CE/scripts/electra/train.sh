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

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

#访问RD程序
cd $code_path
DATA_DIR=$code_path/BookCorpus/

MULTI=$1
MAXSTEPS=$2

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
fi
}

if [[ ${MULTI} == 'multi' ]];then #多卡
    python -u ./run_pretrain.py \
    --model_type electra \
    --model_name_or_path electra-small \
    --input_dir $DATA_DIR \
    --output_dir ./pretrain_model/ \
    --train_batch_size 64 \
    --learning_rate 5e-4 \
    --max_seq_length 128 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 10 \
    --max_steps ${MAXSTEPS} \
    --device gpu  > $log_path/multi_cards_train.log 2>&1
    print_info $? "multi_cards_train"
else #单卡或CPU
    python -u ./run_pretrain.py \
        --model_type electra \
        --model_name_or_path electra-small \
        --input_dir $DATA_DIR \
        --output_dir ./pretrain_model/ \
        --train_batch_size 64 \
        --learning_rate 5e-4 \
        --max_seq_length 128 \
        --weight_decay 1e-2 \
        --adam_epsilon 1e-6 \
        --warmup_steps 10000 \
        --num_train_epochs 1 \
        --logging_steps 1 \
        --save_steps 10 \
        --device gpu \
        --max_steps ${MAXSTEPS} > $log_path/single_card_train.log 2>&1
    print_info $? "single_card_train"
fi
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
