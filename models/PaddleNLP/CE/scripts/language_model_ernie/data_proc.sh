#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/ernie/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
#数据处理逻辑
cd $code_path

wget https://paddlenlp.bj.bcebos.com/data/ernie_hybrid_parallelism_data.tar
tar -xvf ernie_hybrid_parallelism_data.tar

# 下载Vocab文件
wget https://paddlenlp.bj.bcebos.com/data/ernie_hybrid_parallelism-30k-clean.vocab.txt -O ./config/vocab.txt
