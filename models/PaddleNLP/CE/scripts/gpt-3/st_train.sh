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
code_path=$cur_path/../../models_repo/examples/language_model/gpt-3/
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

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0
#访问RD程序
cd $code_path/static
if [[ $2 == "single" ]]; then
  python -u  -m paddle.distributed.fleet.launch \
    --gpus $3 \
    --log_dir "output/gpt3_static/log" run_pretrain_static.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en" \
    --input_dir "./data" \
    --output_dir "output/gpt3_static" \
    --max_seq_len 1024 \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --sharding_degree 1\
    --mp_degree 1 \
    --dp_degree 1 \
    --pp_degree 1 \
    --use_sharding true \
    --use_amp true \
    --use_recompute true \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps 100 \
    --save_steps 20 \
    --decay_steps 320000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 10\
    --eval_freq 10000 \
    --device $1 > $log_path/train_st_$2_$1.log 2>&1
  print_info $? train_st_$2_$1
else
  python -u  -m paddle.distributed.fleet.launch \
    --gpus $3 \
    --log_dir "output/gpt3_static/log" run_pretrain_static.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en" \
    --input_dir "./data" \
    --output_dir "output/gpt3_static" \
    --max_seq_len 1024 \
    --micro_batch_size 8 \
    --global_batch_size 8 \
    --sharding_degree 1\
    --mp_degree 2 \
    --dp_degree 1 \
    --pp_degree 1 \
    --use_sharding true \
    --use_amp true \
    --use_recompute true \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps 100 \
    --save_steps 20 \
    --decay_steps 320000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 10\
    --eval_freq 10000 \
    --device $1 > $log_path/train_st_$2_$1.log 2>&1
  print_info $? train_st_$2_$1
fi
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
