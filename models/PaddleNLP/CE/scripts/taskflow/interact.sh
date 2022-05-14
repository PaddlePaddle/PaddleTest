#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
root_path=$cur_path/../../
log_path=$root_path/log/taskflow/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

print_info(){
cat ${log_path}/$2.log
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

python interact.py > $log_path/train_gpu.log 2>&1

print_info $? train_gpu