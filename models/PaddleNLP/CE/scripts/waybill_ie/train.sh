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
code_path=${nlp_dir}/examples/information_extraction/waybill_ie/

#访问RD程序
cd $code_path

DEVICE=$1
MODEL=$2
if [[ ${MODEL} == "bigru_crf" ]]
then
    python run_bigru_crf.py
elif [[ ${MODEL} == "ernie" ]]
then
    python run_ernie.py
elif [[ ${MODEL} == "ernie_crf" ]]
then
    python run_ernie_crf.py
fi
