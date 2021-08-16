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

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_generation/couplet/
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
cd $code_path

if [[ $1 == "gpu" ]]; then
    python predict.py \
        --num_layers 2 \
        --hidden_size 512 \
        --batch_size 128 \
        --init_from_ckpt couplet_models/final \
        --infer_output_file infer_output.txt \
        --beam_size 10 \
        --device $1 > $log_path/infer_$1.log 2>&1

    print_info $? infer_$1
fi
