#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_generation/ernie-gen/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

print_info(){
cat ${log_path}/$2.log
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

cd $code_path

if [[ $2 == 'multi' ]];then #多卡
    python -m paddle.distributed.launch --gpus $3 ./train.py \
        --model_name_or_path ernie-1.0 \
        --max_encode_len 24 \
        --max_decode_len 72 \
        --batch_size 48  \
        --learning_rate 2e-5 \
        --num_epochs 1 \
        --logging_steps 10 \
        --max_steps 100 \
        --save_steps 10 \
        --output_dir ./tmp/$2/ \
        --device $1 > $log_path/train_$2_$1.log 2>&1
else #单卡
    python -u ./train.py \
        --model_name_or_path ernie-1.0 \
        --max_encode_len 24 \
        --max_decode_len 72 \
        --batch_size 48  \
        --learning_rate 2e-5 \
        --num_epochs 1 \
        --logging_steps 10 \
        --max_steps 100 \
        --save_steps 10 \
        --output_dir ./tmp/$2/ \
        --device $1 > $log_path/train_$2_$1.log 2>&1

fi
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
