#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型语义匹配训练阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/model_zoo/ernie-doc/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi


#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2
CUDA=$3
MODELNAME=$4
TASKNAME=$5


print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}


if [[ ${MULTI} == "single" ]]; then
    python run_semantic_matching.py  \
        --batch_size 6\
        --learning_rate 2e-5 \
        --model_name_or_path ${MODELNAME} \
        --dataset ${TASKNAME}\
        --save_steps 30 \
        --max_steps 30 \
        --logging_steps 1\
        --device ${DEVICE} > $log_path/train_${MULTI}_${TASKNAME}_${DEVICE}.log 2>&1
    print_info $? train_${MULTI}_${TASKNAME}_${DEVICE}
else
    python -m paddle.distributed.launch --gpus ${CUDA} --log_dir cail run_semantic_matching.py\
        --batch_size 6 \
        --learning_rate 2e-5\
        --model_name_or_path ${MODELNAME}\
        --dataset ${TASKNAME}\
        --save_steps 30 \
        --max_steps 30 \
        --logging_steps 1\
        --device ${DEVICE} > $log_path/train_${MULTI}_${TASKNAME}_${DEVICE}.log 2>&1
    print_info $? train_${MULTI}_${TASKNAME}_${DEVICE}
fi

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
