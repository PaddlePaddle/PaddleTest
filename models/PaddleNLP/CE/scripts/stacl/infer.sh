cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型预测阶段"
#路径配置
root_path=$cur_path/../../
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

#访问RD程序
cd $root_path/models_repo/examples/simultaneous_translation/stacl

python predict.py --config ./config/transformer.yaml > $log_path/infer_$2_$1.log 2>&1

print_info $? infer_$2_$1
