#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/model_compression/pp-minilm/general_distill

#准备数据
cd $code_path

cp -r /workspace/task/datasets/PP-MiniLM/*  ./
