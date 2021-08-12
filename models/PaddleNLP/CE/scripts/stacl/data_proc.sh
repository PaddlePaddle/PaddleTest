#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/simultaneous_translation/$model_name
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
#数据处理逻辑
cd $code_path
python -m pip install -r requirements.txt
# 创建数据集目录
rm -rf data/nist2m
mkdir -p data/nist2m

cd data/nist2m
# 待提供数据集bos地址
