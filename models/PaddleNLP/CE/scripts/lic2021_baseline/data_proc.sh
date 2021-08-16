
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
ln -s /workspace/task/datasets/lic2021_baseline/datasets  /workspace/task/models_repo/examples/dialogue/lic2021_baseline/
