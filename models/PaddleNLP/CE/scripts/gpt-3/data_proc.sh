
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/gpt-3
modle_path=$cur_path/../../models_repo/

cd $code_path

cd static

mkdir data && cd data
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
