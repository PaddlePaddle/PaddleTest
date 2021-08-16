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
code_path=$cur_path/../../models_repo/examples/text_classification/pretrained_models/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

#访问RD程序
cd $code_path

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
fi
}

if [[ $1 == 'gpu' ]];then #GPU
    python -m paddle.distributed.launch train.py\
        --batch_size 8\
        --epochs 1 \
        --save_dir ./checkpoints\
        --device $1 > $log_path/train_$2_$1.log 2>&1
    print_info $? train_$2_$1
else #CPU
    python -m train.py\
        --batch_size 8\
        --epochs 1 \
        --save_dir ./checkpoints\
        --device $1 > $log_path/train_$1.log 2>&1
    print_info $? train_$1
fi

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
