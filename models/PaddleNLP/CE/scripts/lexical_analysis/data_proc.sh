
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
code_path=${nlp_dir}/examples/lexical_analysis/
#临时环境更改
cd $code_path
#获取数据&模型逻辑
python download.py --data_dir ./
#数据处理逻辑
