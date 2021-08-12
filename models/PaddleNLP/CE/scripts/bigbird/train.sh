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
code_path=$cur_path/../../models_repo/examples/language_model/$model_name/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

#访问RD程序
cd $code_path
unset CUDA_VISIBLE_DEVICES

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

if [[ $1 == 'gpu' ]];then #GPU
    python -m paddle.distributed.launch --gpus "$4" --log_dir log  run_pretrain.py \
    --model_name_or_path $2 \
    --input_dir "./data" \
    --output_dir "output_multi" \
    --batch_size 4 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --max_steps 10 \
    --save_steps 10 \
    --logging_steps 1 \
    --max_encoder_length 512 \
    --max_pred_length 75 \
    --device $1 > $log_path/train_$3_$1.log 2>&1
    print_info $? train_$3_$1
else #cpu
    python run_pretrain.py --model_name_or_path $2 \
    --input_dir "./data" \
    --output_dir "cpu_output" \
    --batch_size 4 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --max_steps 10 \
    --save_steps 10 \
    --logging_steps 1 \
    --max_encoder_length 512 \
    --max_pred_length 75 \
    --device $1 > $log_path/train_$3_$1.log 2>&1
    print_info $? train_$3_$1
fi

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
