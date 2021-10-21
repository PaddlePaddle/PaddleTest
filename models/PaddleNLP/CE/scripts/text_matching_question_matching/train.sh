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
code_path=$cur_path/../../models_repo/examples/text_matching/question_matching
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

#访问RD程序
cd $code_path

if [[ $2 == "multi" ]]; then
    python -u -m paddle.distributed.launch --gpus $3 train.py \
       --train_set train.txt \
       --dev_set dev.txt \
       --eval_step 50 \
       --save_dir ./checkpoints/$2 \
       --train_batch_size 32 \
       --learning_rate 2E-5 \
       --epochs 1 \
       --save_step 20 \
       --max_steps 100 \
       --rdrop_coef 0.0 \
       --device $1 > $log_path/train_$2_$1.log 2>&1
    print_info $? train_$2_$1
else
    python train.py \
       --train_set train.txt \
       --dev_set dev.txt \
       --eval_step 50 \
       --save_dir ./checkpoints/$2 \
       --train_batch_size 32 \
       --learning_rate 2E-5 \
       --epochs 1 \
       --save_step 20 \
       --max_steps 100 \
       --rdrop_coef 0.0 \
       --device $1  >$log_path/train_$2_$1.log 2>&1
    print_info $? train_$2_$1
fi

#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
