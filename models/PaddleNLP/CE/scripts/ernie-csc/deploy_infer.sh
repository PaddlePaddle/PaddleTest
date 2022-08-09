#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
code_path=${nlp_dir}/examples/text_correction/ernie-csc/


cd $code_path

python export_model.py --params_path checkpoints/multi/best_model.pdparams --output_path ./infer_model/static_graph_params

python predict.py --model_file infer_model/static_graph_params.pdmodel --params_file infer_model/static_graph_params.pdiparams
