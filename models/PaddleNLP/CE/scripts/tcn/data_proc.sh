
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/time_series/$model_name
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
#数据处理逻辑
cd $code_path
#删除之前的文件
rm -rf time_series_covid19_confirmed_global.csv*
#获取数据逻辑
wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
