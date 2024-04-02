#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/model_compression/minilmv2/
#临时环境更改
cd $root_path/models_repo && ls
cd $code_path
#获取数据逻辑

HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

if [ ! -d "10w" ]
then
    wget https://paddlenlp.bj.bcebos.com/models/general_distill/minilmv2_6l_768d_ch.tar.gz
    tar -zxf minilmv2_6l_768d_ch.tar.gz
fi

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
