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
code_path=$cur_path/../../models_repo/examples/text_generation/couplet/
log_path=$root_path/log/$model_name/
mkdir -p $log_path

#访问RD程序
cd $code_path

if [[ $1 == "gpu" ]]; then
    python train.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --device $1 \
    --model_path ./couplet_models \
    --max_epoch 1 > $log_path/train_$2_$1.log 2>&1
fi
