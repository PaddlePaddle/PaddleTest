#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/dialogue/plato-2/
#临时环境更改
cd $root_path/models_repo && ls
cd $code_path

wget https://paddlenlp.bj.bcebos.com/models/transformers/plato2/24L.pdparams

wget https://paddlenlp.bj.bcebos.com/models/transformers/plato2/data.tar.gz
tar -zxf data.tar.gz
