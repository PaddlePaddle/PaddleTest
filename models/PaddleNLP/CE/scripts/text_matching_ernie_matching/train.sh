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
code_path=$cur_path/../../models_repo/examples/text_matching/ernie_matching/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
#访问RD程序
cd $code_path

if [[ $4 == "point-wise" ]]; then
    python -u -m paddle.distributed.launch --gpus $3 train_pointwise.py \
        --device $1 \
        --save_dir ./checkpoints/$4/$2 \
        --batch_size 32 \
        --epochs 1 \
        --save_step 1000 \
        --learning_rate 2E-5 >$log_path/train_$4_$2_$1.log 2>&1

else
    python -u -m paddle.distributed.launch --gpus $3 train_pairwise.py \
        --device $1 \
        --save_dir ./checkpoints/$4/$2 \
        --batch_size 32 \
        --epochs 1 \
        --save_step 1000 \
        --learning_rate 2E-5 >$log_path/train_$4_$2_$1.log 2>&1
fi



#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
