#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/few_shot/p-tuning/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

cd $code_path
  python -u -m paddle.distributed.launch --gpus $3 \
      ptuning.py \
      --task_name $4 \
      --device $1 \
      --p_embedding_num 1 \
      --save_dir "checkpoints/$4/$2" \
      --batch_size 8 \
      --learning_rate 5E-5 \
      --epochs 1 \
      --save_steps 20 \
      --max_seq_length 512 > ${log_path}/train_$4_$2_$1.log 2>&1
