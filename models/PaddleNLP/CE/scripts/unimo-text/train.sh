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
code_path=$cur_path/../../models_repo/examples/text_generation/unimo-text
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
rm -rf ./log/endpoints.log
python -m paddle.distributed.launch --gpus $3 --log_dir ./log run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=unimo-text-1.0 \
    --save_dir=./unimo/checkpoints/$2 \
    --logging_steps=100 \
    --save_steps=100000 \
    --epochs=1 \
    --batch_size=16 \
    --learning_rate=5e-5 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_train \
    --do_predict \
    --device=$1 > $log_path/train_$2_$1.log 2>&1
print_info $? train_$2_$1

#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
