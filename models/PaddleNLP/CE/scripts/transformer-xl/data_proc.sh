
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name

#获取数据逻辑
mkdir -p $code_path/data/

#数据处理逻辑
cd $code_path
# 创建数据集路径
mkdir -p gen_data/
#提前下载好数据集
cp -r /workspace/task/datasets/transformer-xl/gen_data/*  ./gen_data/
# if [ ! -d "gen_data" ]
# then
#     sed -i "s/python3 prep_enwik8.py/python3.7 prep_enwik8.py/g"  ./gen_data.sh
#     bash gen_data.sh
# fi
