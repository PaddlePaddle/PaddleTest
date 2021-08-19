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
code_path=$cur_path/../../models_repo/examples/language_model/ernie/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

DEVICE=$1
if [[ ${DEVICE} == "gpu" ]]; then
N_GPU=1
else
N_GPU=0
fi
MULTI=$2
if [[ ${MULTI} == "multi" ]]; then
N_GPU=2
fi

#访问RD程序
cd $code_path

if [[ $4 == 'con' ]]; then
  NUM_STEPS=1000000
  SAVE_STEPS=100000
else
  NUM_STEPS=100
  SAVE_STEPS=10
fi

mkdir -p output_dir/${MULTI}

if [[ ${MULTI} == "single" ]]; then
    python -m paddle.distributed.fleet.launch \
        --gpus $3 \
        --log_dir ./output_dir/log \
        run_pretraining.py \
        --global_bsz 64 \
        --micro_bsz 1 \
        --max_seq_len 512 \
        --ernie_config_file config/ernie_base_config.json \
        --learning_rate 1e-4 \
        --log_steps 1 \
        --num_train_steps ${NUM_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --output_dir ./output_dir/${MULTI} \
        --use_recompute true \
        --use_sharding true \
        --use_sop false \
        --num_mp=1 \
        --num_sharding=$N_GPU \
        --num_pp=1 \
        --num_dp=1  >$log_path/train_${MULTI}_${DEVICE}.log 2>&1

else
    python -m paddle.distributed.fleet.launch \
        --gpus $3 \
        --log_dir ./output_dir/log \
        run_pretraining.py \
        --global_bsz 64 \
        --micro_bsz 1 \
        --max_seq_len 512 \
        --ernie_config_file config/ernie_base_config.json \
        --learning_rate 1e-4 \
        --log_steps 1 \
        --num_train_steps ${NUM_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --output_dir ./output_dir/${MULTI} \
        --use_recompute true \
        --use_sharding true \
        --use_sop false \
        --num_mp=1 \
        --num_sharding=$N_GPU \
        --num_pp=1 \
        --num_dp=1  >$log_path/train_${MULTI}_${DEVICE}.log 2>&1

fi

#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
