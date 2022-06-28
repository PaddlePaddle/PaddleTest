
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
code_path=${nlp_dir}/examples/text_matching/simbert/
data_path=$1
#准备数据
cd $code_path

cp -r ${data_path}/*  ./
