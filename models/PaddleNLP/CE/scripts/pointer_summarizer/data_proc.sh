
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
modle_path=$cur_path/../../models_repo/
code_path=$cur_path/../../models_repo/examples/text_summarization/pointer_summarizer
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

# 准备数据
cd $code_path
cp -r /workspace/task/datasets/pointer_summarizer/*  ./
