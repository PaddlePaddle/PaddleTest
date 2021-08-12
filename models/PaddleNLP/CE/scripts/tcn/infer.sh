cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型样例测试阶段"
#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/time_series/$model_name
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#访问RD程序
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
if [[ $1 == "gpu" ]]; then
  python predict.py --data_path time_series_covid19_confirmed_global.csv \
      --use_gpu > $log_path/infer_$1.log 2>&1
  print_info $? infer_$1
else
  python predict.py --data_path time_series_covid19_confirmed_global.csv > $log_path/infer_$1.log 2>&1
  print_info $? infer_$1
fi
