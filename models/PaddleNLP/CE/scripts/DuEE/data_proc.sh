
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
code_path=$cur_path/../../models_repo/examples/information_extraction/DuEE
log_path=$root_path/log/$model_name/
data_path=$code_path/data/DuEE-Fin
conf_path=$code_path/conf/DuEE-Fin
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

# 准备数据
#配置目标数据存储路径
rm -rf $data_path
rm -rf $conf_path
mkdir -p $data_path
mkdir -p $conf_path
#准备数据
cd $code_path
cp -r /workspace/task/datasets/DuEE/*  ./data/DuEE-Fin/
# 拷贝数据到conf下
cp -r /workspace/task/datasets/DuEE/*  ./conf/DuEE-Fin/
# 覆盖bug
sed -i 's/title \= doc\[\"title\"\]/title \= doc\.get\(\"title\", \"\"\)/g'  ./duee_fin_data_prepare.py
bash run_duee_fin.sh data_prepare
