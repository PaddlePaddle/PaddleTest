cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型样例测试阶段"
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
cd examples/language_model/gpt

python export_model.py --model_type=gpt \
    --model_path=gpt2-medium-en\
    --output_path=./infer_model/model

python deploy/python/inference.py --model_type gpt \
    --model_path ./infer_model/model > $log_path/infer.log 2>&1

print_info $? "infer"
#cat $log_path/infer.log
