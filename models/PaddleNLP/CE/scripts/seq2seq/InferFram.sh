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

echo "$model_name 模型样例测试阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/machine_translation/seq2seq/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/models_repo

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

DEVICE=$1

python export_model.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --init_from_ckpt attention_models/final.pdparams \
    --beam_size 10 \
    --export_path ./Infer_model/model >$log_path/inferfram_${DEVICE}.log 2>&1

print_info $? inferfram_${DEVICE}

cd deploy/python
python3.7 infer.py \
    --export_path ../../Infer_model/model \
    --device ${DEVICE} \
    --batch_size 128 \
    --infer_output_file Infer_output.txt >>$log_path/inferfram_${DEVICE}.log 2>&1

print_info $? inferfram_${DEVICE}
cd -


#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
