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
code_path=$cur_path/../../models_repo/examples/text_classification/rnn/
log_path=$root_path/log/$model_name/
mkdir -p $log_path

#访问RD程序
cd $code_path

DEVICE=$1

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

python export_model.py \
  --vocab_path=./vocab.json \
  --network=bilstm \
  --params_path=./checkpoints/final.pdparams \
  --output_path=./static_graph_params

python deploy/python/predict.py \
  --model_file=static_graph_params.pdmodel \
  --params_file=static_graph_params.pdiparams \
  --network=bilstm \
  --device=${DEVICE} > $log_path/python_infer_${DEVICE}.log 2>&1

print_info $? python_infer_${DEVICE}


#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
