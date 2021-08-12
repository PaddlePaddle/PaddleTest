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
code_path=$cur_path/../../models_repo/examples/semantic_indexing/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0



print_info(){
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

#访问RD程序
cd $code_path

if [[ $4 == "batch" ]]; then
    python -u -m paddle.distributed.launch --gpus "$3" train_batch_neg.py \
        --device $1 \
        --save_dir ./checkpoints_batch_neg_$2/ \
        --batch_size 128 \
        --learning_rate 5E-5 \
        --epochs 1 \
        --output_emb_size 256 \
        --save_steps 1000 \
        --max_seq_length 64 \
        --margin 0.2 \
        --train_set_file semantic_pair_train.tsv > ${log_path}/train_$4_$2_$1.log 2>&1
else
    python -u -m paddle.distributed.launch --gpus "$3" train_hardest_neg.py \
        --device $1\
        --save_dir ./checkpoints_hardest_neg_$2/ \
        --batch_size 128 \
        --learning_rate 5E-5 \
        --epochs 1 \
        --output_emb_size 256 \
        --save_steps 1000 \
        --max_seq_length 64 \
        --margin 0.2 \
        --train_set_file semantic_pair_train.tsv > ${log_path}/train_$4_$2_$1.log 2>&1
fi

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
