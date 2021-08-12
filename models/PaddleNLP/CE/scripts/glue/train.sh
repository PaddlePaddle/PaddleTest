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
code_path=$cur_path/../../models_repo/examples/benchmark/$model_name/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi


#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2
MODELTYPE=$3
MODELNAME=$4
TASKNAME=$5
LR=$6

if [[ ${TASKNAME} == 'CoLA' ]];then
    max_steps=100
else
    max_steps=10
fi

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
fi
}

python -u ./run_glue.py\
        --model_type ${MODELTYPE}\
        --model_name_or_path ${MODELNAME} \
        --task_name ${TASKNAME} \
        --max_seq_length 128 \
        --batch_size 32  \
        --learning_rate ${LR} \
        --num_train_epochs 1  \
        --logging_steps 1 \
        --save_steps 10\
        --output_dir ./$3/\
        --max_steps $max_steps\
        --device $DEVICE > $log_path/train_${MODELNAME}_${TASKNAME}_${MULTI}_${DEVICE}.log 2>&1

print_info $? train_${MODELNAME}_${TASKNAME}_${MULTI}_${DEVICE}

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
