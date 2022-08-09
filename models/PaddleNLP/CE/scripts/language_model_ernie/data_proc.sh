#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
code_path=${nlp_dir}/model_zoo/ernie-1.0/

#数据处理逻辑
cd $code_path
mkdir data && cd data
wget https://paddlenlp.bj.bcebos.com/models/transformers/data_tools/ernie_wudao_0903_92M_ids.npy
wget https://paddlenlp.bj.bcebos.com/models/transformers/data_tools/ernie_wudao_0903_92M_idx.npz
