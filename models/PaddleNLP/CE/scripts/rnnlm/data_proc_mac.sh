#unset http_proxy
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/rnnlm/
dataset_path=/Users/paddle/.paddlenlp/datasets/PTB/simple-examples
#临时环境更改
cd $root_path/models_repo && ls
cd $code_path
#获取数据逻辑，将数据准备好

if [ ! -d $dataset_path ]; then
  mkdir -p $dataset_path
fi
# mac 的拷贝
cp -r /Users/paddle/ce_data/PaddleNLP/rnnlm/simple-examples/*  /Users/paddle/.paddlenlp/datasets/PTB/simple-examples/
#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
