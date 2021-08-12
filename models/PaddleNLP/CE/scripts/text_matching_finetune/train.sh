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
code_path=$cur_path/../../models_repo/examples/text_matching/sentence_transformers/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#访问RD程序
cd $code_path

DEVICE=$1

# 控制保存模型
sed -i "s/if global_step % 100 == 0 and rank == 0:/if global_step % 1000 == 0 and rank == 0:/g" train.py

python -m paddle.distributed.launch --gpus $3 train.py \
  --device ${DEVICE} \
  --save_dir ./checkpoints  \
  --epochs 1 >$log_path/train_$2_${DEVICE}.log 2>&1



#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
