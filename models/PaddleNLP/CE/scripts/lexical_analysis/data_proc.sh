
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/lexical_analysis/
#临时环境更改
cd $root_path/models_repo && ls
cd $code_path
#获取数据&模型逻辑
python download.py --data_dir ./
#数据处理逻辑
