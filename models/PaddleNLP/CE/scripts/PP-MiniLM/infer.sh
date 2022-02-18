cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型预测阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/model_compression/pp-minilm/deploy/python
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
NAME=$(echo $3 | tr 'A-Z' 'a-z')
LR=$4
BS=$5
MAX_SEQ_LEN=$6

#访问RD程序
cd $code_path
python infer.py --task_name ${NAME}  --model_path  ../../quantization/${NAME}_quant_models/mse4/int8  --int8 --collect_shape
python infer.py --task_name ${NAME}  --model_path  ../../quantization/${NAME}_quant_models/mse4/int8  --int8 > ${log_path}/infer_${TASK}_${LR}_${BS}_${1}.log 2>&1
print_info $? infer_${TASK}_${LR}_${BS}_${1}
