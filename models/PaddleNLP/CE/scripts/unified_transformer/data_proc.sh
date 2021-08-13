
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径 [用户改]
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/dialogue/unified_transformer/

#获取数据逻辑 [用户改]
cd $code_path
