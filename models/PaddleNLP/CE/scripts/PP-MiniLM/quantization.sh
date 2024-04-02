cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型量化阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/model_compression/pp-minilm/
output_path=$cur_path/../../models_repo/examples/model_compression/pp-minilm/pruning/pruned_models
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

TASK=$3
LR=$4
BS=$5
MAX_SEQ_LEN=$6

#访问RD程序
cd $code_path/quantization
python quant_post.py --task_name $TASK --input_dir ${output_path}/${TASK}/0.75/sub_static > ${log_path}/quantization_${TASK}_${LR}_${BS}_$1.log 2>&1
print_info $? quantization_${TASK}_${LR}_${BS}_$1
