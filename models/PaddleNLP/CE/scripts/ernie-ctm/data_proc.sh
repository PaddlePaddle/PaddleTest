#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
code_path=${nlp_dir}/examples/text_to_knowledge/ernie-ctm/
cd $code_path
#获取数据&模型逻辑
#数据处理逻辑
wget https://paddlenlp.bj.bcebos.com/paddlenlp/datasets/wordtag_dataset_v2.tar.gz
tar -zxvf wordtag_dataset_v2.tar.gz
