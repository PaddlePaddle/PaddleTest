#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
code_path=${nlp_dir}/examples/text_correction/ernie-csc/


cd $code_path

python sighan_evaluate.py -p predict_sighan13.txt -t sighan_test/sighan13/truth.txt
