#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型预测阶段"


#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_generation/ernie-gen/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

print_info(){
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

cd $code_path
python -u ./predict.py \
    --model_name_or_path ernie-1.0 \
    --max_encode_len 24 \
    --max_decode_len 72 \
    --batch_size 48   \
    --init_checkpoint ./tmp/model_10000/model_state.pdparams \
    --device $1 > $log_path/infer_$1.log 2>&1

print_info $? infer_$1