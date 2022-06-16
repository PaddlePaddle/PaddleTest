cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型预测阶段"
#路径配置

#访问RD程序
cd ${nlp_dir}/examples/simultaneous_translation/stacl
python predict.py --config ./config/transformer.yaml

