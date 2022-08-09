cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型Fine-tune阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/model_zoo/$model_name/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

print_info(){
  cat ${log_path}/$2.log
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

cd $code_path
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus $2 run_glue.py\
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name $1 \
  --max_seq_length 128 \
  --batch_size 32   \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --save_steps 10 \
  --output_dir ./$1/ \
  --device gpu \
  --max_steps 20\
  --use_amp False > $log_path/$1_$3_fine-tune.log 2>&1

print_info $? $1_$3_fine-tune
