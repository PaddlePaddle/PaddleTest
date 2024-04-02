#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
code_path=${nlp_dir}/examples/simultaneous_translation/$model_name

#数据处理逻辑
cd $code_path
python -m pip install -r requirements.txt
# 创建数据集目录
rm -rf data/nist2m
mkdir -p data/nist2m

cp -r  /workspace/task/datasets/stacl/data/nist2m/*  ./data/nist2m/
