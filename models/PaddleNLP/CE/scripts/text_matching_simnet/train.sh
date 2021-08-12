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
code_path=$cur_path/../../models_repo/examples/text_matching/simnet/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#访问RD程序
cd $code_path

DEVICE=$1
if [[ ${DEVICE} == "gpu" ]]; then
  python -m paddle.distributed.launch --gpus "$3" train.py \
    --vocab_path='./simnet_vocab.txt' \
    --device=${DEVICE} \
    --network=lstm \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=1 \
    --save_dir='./checkpoints' > $log_path/train_$2_${DEVICE}.log 2>&1
else
  python train.py --vocab_path='./simnet_vocab.txt' \
   --device=${DEVICE} \
   --network=lstm \
   --lr=5e-4 \
   --batch_size=64 \
   --epochs=5 \
   --save_dir='./checkpoints' > $log_path/train_${DEVICE}.log 2>&1
fi


#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
