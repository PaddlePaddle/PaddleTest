#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/time_series/$model_name
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
#访问RD程序
cd $code_path

if [ $1 == 'gpu' ];then #GPU
    python train.py --data_path time_series_covid19_confirmed_global.csv \
                --epochs 1 \
                --batch_size 32 \
                --use_gpu > $log_path/train_$2_$1.log 2>&1
    print_info $? train_$2_$1
# elif [[ $1 == 'recv' ]];then
# recv

else #CPU
    python train.py --data_path time_series_covid19_confirmed_global.csv \
                --epochs 1 \
                --batch_size 32 > $log_path/train_$1.log 2>&1
    print_info $? train_$1
fi
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
