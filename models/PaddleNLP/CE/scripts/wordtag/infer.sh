#unset http_proxy
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型预测阶段阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/text_to_knowledge/wordtag/
log_path=$root_path/log/$model_name/

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

python predict.py --max_seq_len 128 --batch_size 2 --device $1 > $log_path/infer_$1.log 2>&1

print_info $? infer_$1