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
code_path=$cur_path/../../models_repo/examples/text_classification/pretrained_models/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

#访问RD程序
cd $code_path

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

if [[ $1 == 'gpu' ]];then #GPU
    python predict.py\
        --device $1\
        --params_path checkpoints/model_900/model_state.pdparams\
        > $log_path/infer_$2_$1.log 2>&1
    print_info $? infer_$2_$1
else #CPU 跟GPU只是log后缀不一样，也可以传两个参数保持跟GPU一直，这里就可以省掉
    python predict.py\
        --device $1\
        --params_path checkpoints/model_900/model_state.pdparams\
        > $log_path/infer_$1.log 2>&1
    print_info $? infer_$1
fi

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
