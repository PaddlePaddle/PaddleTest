
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_graph/erniesage/

#数据处理逻辑
cd $code_path
mkdir -p graph_workdir
python ./preprocessing/dump_graph.py --conf ./config/erniesage_link_prediction.yaml
