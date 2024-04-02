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
code_path=$cur_path/../../models_repo/model_zoo/ernie-health/cblue/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi


#访问RD程序
cd $code_path
unset CUDA_VISIBLE_DEVICES

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

if [[ $1 == 'gpu' ]];then #GPU
    python -m paddle.distributed.launch --gpus $3 train_ner.py \
        --batch_size 32 \
        --max_seq_length 128 \
        --learning_rate 6e-5 \
        --epochs 1 \
        --max_steps 20 \
        --save_steps 10 \
        --logging_steps 10 \
        --valid_steps 10 \
        --save_dir ./checkpoint/CMeEE/$2 > $log_path/identify_$2_$1.log 2>&1
    print_info $? identify_$2_$1
fi

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
