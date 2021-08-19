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

echo "$model_name 模型样例测试阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/information_extraction/msra_ner/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/models_repo

#访问RD程序
cd $code_path

DEVICE=$1
if [[ ${DEVICE} == "gpu" ]]; then
USE_GPU=1
else
USE_GPU=0
fi

python -u ./eval.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 32 \
    --device ${DEVICE} \
    --init_checkpoint_path tmp/msra_ner/model_500.pdparams > $log_path/eval_${DEVICE}.log 2>&1

cat $log_path/eval_${DEVICE}.log

#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
