
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/applications/sentiment_analysis

#数据处理逻辑
cd $code_path
mkdir -p checkpoints/ext_checkpoints
mkdir -p checkpoints/cls_checkpoints
mkdir -p checkpoints/pp_checkpoints
mkdir data
cd data
wget https://bj.bcebos.com/v1/paddlenlp/data/ext_data.tar.gz
tar -xzvf ext_data.tar.gz
wget https://bj.bcebos.com/v1/paddlenlp/data/cls_data.tar.gz
tar -xzvf cls_data.tar.gz
