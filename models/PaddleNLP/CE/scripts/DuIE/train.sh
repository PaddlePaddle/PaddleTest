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
code_path=$cur_path/../../models_repo/examples/information_extraction/DuIE/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi


echo $CUDA_VISIBLE_DEVICES

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}


cd $code_path
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus $3 run_duie.py \
    --device $1 \
    --seed 42 \
    --do_train \
    --data_path ./data \
    --max_seq_length 128 \
    --batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 2e-5\
    --warmup_ratio 0.06 \
    --output_dir ./checkpoints/$2 > $log_path/train_$2_$1.log 2>&1
