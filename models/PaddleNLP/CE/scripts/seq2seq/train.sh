#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/machine_translation/seq2seq/

#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2

if [[ ${DEVICE} == "gpu" ]]; then
    python train.py \
        --num_layers 2 \
        --hidden_size 512 \
        --batch_size 128 \
        --max_epoch 1 \
        --log_freq 1 \
        --dropout 0.2 \
        --init_scale  0.1 \
        --max_grad_norm 5.0 \
        --device ${DEVICE} \
        --model_path ./attention_models

fi
