
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/dialogue/$model_name
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy
#数据处理逻辑
cd $code_path

wget https://paddlenlp.bj.bcebos.com/datasets/DGU_datasets.tar.gz
tar -zxf DGU_datasets.tar.gz

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
