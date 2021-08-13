
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径 [用户改]
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/machine_translation/transformer

#获取数据逻辑 [用户改]
cd $code_path
wget http://gitlab.baidu.com/liuzhengxi/PaddleNLP/blob/qa/examples/machine_translation/transformer/gen_data.sh
#数据处理逻辑
./gen_data.sh

ls gen_data
