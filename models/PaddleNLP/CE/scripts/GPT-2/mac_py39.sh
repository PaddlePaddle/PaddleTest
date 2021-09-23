
#获取当前路径
cur_path=`pwd`
#配置目标数据存储路径
code_path=$cur_path/../../models_repo/examples/language_model/data_tools/
cd $code_path
sed -i '' 's/CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color/CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color -Wl,-undefined,dynamic_lookup/g'  Makefile
