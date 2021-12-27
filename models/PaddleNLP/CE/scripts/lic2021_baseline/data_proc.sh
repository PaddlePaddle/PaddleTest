
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/dialogue/lic2021_baseline/
echo "$model_name 模型数据预处理阶段"

cd $code_path
#配置目标数据存储路径
cp -r /workspace/task/datasets/lic2021_baseline/datasets  ./
