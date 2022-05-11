#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_correction/ernie-csc/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

print_info(){
cat ${log_path}/$2.log
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

cd $code_path

if [[ $2 == 'single' ]];then #多卡
    python train.py \
        --batch_size 32 \
        --logging_steps 10 \
        --epochs 1 \
        --save_steps 100\
        --max_steps 200 \
        --learning_rate 5e-5 \
        --model_name_or_path ernie-1.0 \
        --output_dir ./checkpoints/$2 \
        --extra_train_ds_dir ./extra_train_ds/ \
        --max_seq_length 192 \
        --device $1> $log_path/train_$2_$1.log 2>&1
    print_info $? train_$2_$1
else
    python -m paddle.distributed.launch --gpus $3  train.py \
        --batch_size 32 \
        --logging_steps 10 \
        --max_steps 200 \
        --epochs 1 \
        --save_steps 100\
        --learning_rate 5e-5 \
        --model_name_or_path ernie-1.0  \
        --output_dir ./checkpoints/$2 \
        --extra_train_ds_dir ./extra_train_ds/  \
        --max_seq_length 192\
        --device $1 > $log_path/train_$2_$1.log 2>&1
    print_info $? train_$2_$1
fi
