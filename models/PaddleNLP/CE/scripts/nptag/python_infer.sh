
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型预测阶段"

#配置目标数据存储路径
code_path=${nlp_dir}/examples/text_to_knowledge/nptag


# 准备数据
cd $code_path
python export_model.py --params_path=./output/single/model_100/model_state.pdparams --output_path=./export
python deploy/python/predict.py --model_dir=./export
