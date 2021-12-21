cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型样例评估阶段"

#路径配置
root_path=$cur_path/../../
log_path=$root_path/log/$model_name/
mkdir -p $log_path

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

#访问RD程序
cd $root_path/models_repo
cd examples/text_to_knowledge/ernie-ctm


python -m paddle.distributed.launch --gpus $2 predict.py \
    --params_path ./tmp/model_100/model_state.pdparams \
    --batch_size 32 \
    --device $1 > $log_path/infer_$1.log 2>&1

print_info $? infer_$1
