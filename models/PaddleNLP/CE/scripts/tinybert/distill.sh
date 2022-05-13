cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型蒸馏阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/model_zoo/tinybert/
log_path=$root_path/log/$model_name/
data_path=$cur_path/../../models_repo/examples/benchmark/glue/tmp/$3/$2/
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

NAME=$(echo $3 | tr 'A-Z' 'a-z')

python task_distill.py \
    --model_type tinybert \
    --student_model_name_or_path tinybert-6l-768d-v2 \
    --task_name $3 \
    --intermediate_distill \
    --max_seq_length 64 \
    --batch_size 32   \
    --T 1 \
    --teacher_model_type bert \
    --teacher_path $data_path/${NAME}_ft_model_30.pdparams \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 10 \
    --max_steps 30 \
    --output_dir ./tmp/$3/$2 \
    --device $1 > $log_path/distill_$3_$2_$1.log

print_info $? distill_$3_$2_$1

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
