#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/word_embedding/

#访问RD程序
cd $code_path

DEVICE=$1
# 是否使用use_token_embedding
EMBEDDING=$3
if [[ ${EMBEDDING} ]];then
  python train.py \
    --device=${DEVICE} \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=1 \
    --use_token_embedding=True \
    --vdl_dir='./vdl_dir'
else
  python train.py \
    --device=${DEVICE} \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=1 \
    --use_token_embedding=False \
    --vdl_dir='./vdl_dir'
fi
