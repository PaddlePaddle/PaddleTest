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
code_path=$cur_path/../../models_repo/examples/language_model/ernie-1.0/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

DEVICE=$1
#访问RD程序
cd $code_path

if [[ $4 == 'con' ]]; then
  MAX_STEPS=4000000
  SAVE_STEPS=50000
else
  MAX_STEPS=100
  SAVE_STEPS=20
fi

if [[ $2 == 'single' ]]; then
  DP_DEGREE=1
  GLOBAL_BATCH_SIZE=32
else
  DP_DEGREE=2
  GLOBAL_BATCH_SIZE=64
fi

mkdir -p output_dir/${MULTI}

python -u  -m paddle.distributed.launch  \
    --gpus "$3" \
    --log_dir "./log" \
    run_pretrain_static.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0" \
    --input_dir "./data/" \
    --output_dir "./output/" \
    --max_seq_len 512 \
    --micro_batch_size 32 \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --sharding_degree 1 \
    --dp_degree $DP_DEGREE \
    --use_sharding false \
    --use_amp true \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --checkpoint_steps 5000 \
    --decay_steps 3960000 \
    --weight_decay 0.01 \
    --warmup_rate 0.0025 \
    --grad_clip 1.0 \
    --logging_freq 20\
    --num_workers 2 \
    --eval_freq 1000 \
    --device $1 >$log_path/train_${MULTI}_${DEVICE}.log 2>&1
