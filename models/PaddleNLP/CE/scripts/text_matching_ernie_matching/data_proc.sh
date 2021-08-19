
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_matching/ernie_matching/
dataset_path=/root/.paddlenlp/datasets/LCQMC/lcqmc
if [ ! -d $dataset_path ]; then
  mkdir -p $dataset_path
fi

#临时环境更改
cd $code_path
#拷贝数据
cp -r /workspace/task/datasets/ernie_matching/lcqmc/*  /root/.paddlenlp/datasets/LCQMC/lcqmc/
cp -r /workspace/task/datasets/ernie_matching/lcqmc/test.tsv ./
