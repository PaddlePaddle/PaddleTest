
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

code_path=${nlp_dir}/examples/information_extraction/waybill_ie/

cd $code_path
python download.py --data_dir ./waybill_ie
#获取数据&模型逻辑
#数据处理逻辑
