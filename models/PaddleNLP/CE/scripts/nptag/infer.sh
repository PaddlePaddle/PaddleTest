
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型预测阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
modle_path=$cur_path/../../models_repo/
code_path=$cur_path/../../models_repo/examples/text_to_knowledge/nptag
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

# 准备数据
cd $code_path
python -m paddle.distributed.launch --gpus $2 predict.py \
    --device=$1 \
    --params_path ./output/single/model_100/model_state.pdparams > $log_path/infer_$1.log 2>&1
print_info $? infer_$1
