
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/gpt/
modle_path=$cur_path/../../models_repo/
py_path=$code_path/py_gpu_mem.py

# 将统计显卡的脚本拷贝到代码路径下
mv ./py_gpu_mem.py  $py_path

#获取数据逻辑
#清除之前下载的脚本
rm -rf $code_path/data

mkdir -p $code_path/data

wget -P $code_path/data https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/dataset/my-gpt2_text_document_ids.npz
