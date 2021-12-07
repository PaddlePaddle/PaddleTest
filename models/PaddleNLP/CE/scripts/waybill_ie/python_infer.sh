#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型python预测部署阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/information_extraction/waybill_ie/
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

cd $code_path

python export_model.py --params_path $2_ckpt/model_80/model_state.pdparams --output_path=./$2_output > ${log_path}/python_infer_$2_$1.log 2>&1
step_one_code=$?
if [ $step_one_code -ne 0 ];then
    print_info $step_one_code python_infer_$2_$1
else
    python deploy/python/predict.py --model_dir ./$2_output > ${log_path}/python_infer_$2_$1.log 2>&1
    print_info $? python_infer_$2_$1
fi
