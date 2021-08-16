#unset http_proxy
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/machine_translation/seq2seq/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/models_repo

#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
fi
}

if [[ ${DEVICE} == "gpu" ]]; then
    python train.py \
        --num_layers 2 \
        --hidden_size 512 \
        --batch_size 128 \
        --max_epoch 1 \
        --dropout 0.2 \
        --init_scale  0.1 \
        --max_grad_norm 5.0 \
        --device ${DEVICE} \
        --model_path ./attention_models >$log_path/train_${MULTI}_${DEVICE}.log 2>&1

    print_info $? train_${MULTI}_${DEVICE}

fi

#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
