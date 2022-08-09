
#获取当前路径
cur_path=`pwd`
#配置目标数据存储路径
code_path=$cur_path/../../models_repo/model_zoo/ernie-1.0/data_tools/
cd $code_path
sed -i '' 's/python3 -m pybind11/python3.9 -m pybind11/g'  Makefile
sed -i '' 's/python3-config --extension-suffix/python3.9-config --extension-suffix/g' Makefile
sed -i '' 's/CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color/CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color -Wl,-undefined,dynamic_lookup/g'  Makefile
