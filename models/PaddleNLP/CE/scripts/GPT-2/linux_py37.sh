
#获取当前路径
cur_path=`pwd`
#配置目标数据存储路径
code_path=$cur_path/../../models_repo/model_zoo/ernie-1.0/data_tools/
cd $code_path
sed -i 's/python3 -m pybind11/python3.7 -m pybind11/g'  Makefile
sed -i 's/python3-config --extension-suffix/python3.7m-config --extension-suffix/g' Makefile
