
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/word_embedding/
#临时环境更改
cd $root_path/models_repo && ls
cd $code_path
#获取数据逻辑
if [ ! -f "dict.txt" ]
then
    wget https://paddlenlp.bj.bcebos.com/data/dict.txt
fi
#数据处理逻辑
