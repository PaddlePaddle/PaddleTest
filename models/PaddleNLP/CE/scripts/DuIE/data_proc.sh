#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#配置目标数据存储路径
root_path=$cur_path/../../
modle_path=$cur_path/../../models_repo/
code_path=$cur_path/../../models_repo/examples/information_extraction/DuIE/
log_path=$root_path/log/$model_name/
data_path=$code_path/data

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

# 准备数据
#配置目标数据存储路径

cp -r /workspace/task/datasets/DuIE/*  /workspace/task/models_repo/examples/information_extraction/DuIE/data/
cd $code_path
# 替换代码
sed -i "s/python3 .\/re_official_evaluation.py/python .\/re_official_evaluation.py/g"  ./utils.py
