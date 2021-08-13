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
code_path=$cur_path/../../models_repo/examples/text_classification/rnn/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/models_repo

#访问RD程序
cd $code_path

DEVICE=$1

python -m paddle.distributed.launch --gpus $3 train.py \
  --vocab_path='./senta_word_dict.txt' \
  --device=${DEVICE} \
  --network=bilstm \
  --lr=5e-4 \
  --batch_size=64 \
  --epochs=1 \
  --save_dir='./checkpoints' >$log_path/train_$2_${DEVICE}.log 2>&1

#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
