cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型Fine-tune阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
#访问RD程序

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
fi
}

cd $code_path
python -u ./run_glue.py \
    --model_type electra \
    --model_name_or_path electra-small \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 10 \
    --output_dir ./SST-2/ \
    --max_steps 50\
    --n_gpu 1 > $log_path/single_fine-tune.log 2>&1

print_info $? single_fine-tune
