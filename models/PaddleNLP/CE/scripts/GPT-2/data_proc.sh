
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/gpt/
modle_path=$cur_path/../../models_repo/
#初始化一下visualdl
python init.py
#获取数据逻辑
#清除之前下载的脚本
rm -rf $code_path/raw_data
rm -rf $code_path/data

mkdir -p $code_path/data
wget -P $code_path/data https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/train.data.json_ids.npz
