
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

cd $code_path
#获取数据逻辑
mkdir -p $code_path/data
wget -P $code_path/data https://paddlenlp.bj.bcebos.com/ce/data/bigbird/wiki.csv


export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
