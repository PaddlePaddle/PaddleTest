cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型finetune阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/model_compression/minilmv2/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

cd $code_path

python -m paddle.distributed.launch --gpus $3 general_distill.py \
  --student_model_type tinybert \
  --num_relation_heads 48 \
  --student_model_name_or_path tinybert-6l-768d-zh \
  --init_from_student False \
  --teacher_model_type bert \
  --teacher_model_name_or_path bert-base-chinese \
  --max_seq_length 128 \
  --batch_size 256 \
  --learning_rate 6e-4 \
  --logging_steps 10 \
  --max_steps 30 \
  --warmup_steps 4000 \
  --save_steps 10 \
  --teacher_layer_index 11 \
  --student_layer_index 5 \
  --weight_decay 1e-2 \
  --output_dir ./pretrain \
  --device $1 \
  --input_dir ./data > $log_path/finetune_$2_$1.log 2>&1

print_info $? finetune_$2_$1

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
