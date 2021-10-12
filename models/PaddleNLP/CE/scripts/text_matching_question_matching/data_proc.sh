
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_matching/question_matching

#获取数据逻辑
mkdir -p $code_path/data/

#数据处理逻辑
cd $code_path
#提前下载好数据集
cp -r /workspace/task/datasets/question_matching/data_v4/*  ./data/

# 将三个数据集合并成一个
cat ./data/train/LCQMC/train ./data/train/BQ/train ./data/train/OPPO/train > train.txt
cat ./data/train/LCQMC/dev ./data/train/BQ/dev ./data/train/OPPO/dev > dev.txt