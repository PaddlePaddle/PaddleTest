#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型预测动转静阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/machine_translation/$model_name
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
cd $code_path
#动转静
python export_model.py --config ./configs/transformer.base.yaml >> $log_path/dy_to_st_$1.log 2>&1

#完成动转静的静态图的模型, 实现高性能预测的功能
python deploy/python/inference.py \
    --config ./configs/transformer.base.yaml \
    --batch_size 8 \
    --device $1 \
    --model_dir ../../infer_model/ >> $log_path/dy_to_st_$1.log 2>&1
print_info $? dy_to_st_$1
