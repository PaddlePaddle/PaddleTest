#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/semantic_indexing/
#临时环境更改
cd $root_path/models_repo && ls
cd $code_path
#获取数据&模型逻辑
#数据处理逻辑
python -m pip install hnswlib
wget https://paddlenlp.bj.bcebos.com/models/semantic_index/semantic_pair_train.tsv
wget https://paddlenlp.bj.bcebos.com/models/semantic_index/same_semantic.tsv
wget https://paddlenlp.bj.bcebos.com/models/semantic_index/corpus_file
