#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_correction/ernie-csc/
cd $code_path
#获取数据&模型逻辑
python download.py --data_dir ./extra_train_ds/ --url https://github.com/wdimmy/Automatic-Corpus-Generation/raw/master/corpus/train.sgml

# 预处理数据集
python change_sgml_to_txt.py -i extra_train_ds/train.sgml -o extra_train_ds/train.txt

if [ ! -d "./sighan_test" ]; then
  python download.py
fi
