
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
#获取数据逻辑
rm -rf $code_path/BookCorpus/
mkdir -p $code_path/BookCorpus/
#here need bos url
wget -P $code_path/BookCorpus https://paddlenlp.bj.bcebos.com/models/electra/train.data.txt
#cp /home/xishengdai/train.data $code_path/BookCorpus/

cd $code_path/BookCorpus
mv train.data.txt train.data

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
