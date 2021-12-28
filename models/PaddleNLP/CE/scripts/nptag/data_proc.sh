
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
code_path=$cur_path/../../models_repo/examples/text_to_knowledge/nptag

# 准备数据
cd $code_path
wget https://bj.bcebos.com/paddlenlp/paddlenlp/datasets/nptag_dataset.tar.gz && tar -zxvf nptag_dataset.tar.gz
