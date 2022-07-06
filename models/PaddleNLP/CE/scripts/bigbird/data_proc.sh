#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
code_path=${nlp_dir}/examples/language_model/$model_name

cd $code_path
#获取数据逻辑
mkdir -p $code_path/data
wget -P $code_path/data https://paddlenlp.bj.bcebos.com/ce/data/bigbird/wiki.csv
