#exit 0

source ~/.bashrc
pyenv activate PaddleClas
which python
echo "$real_compile_path"
python --version
export Project_path=${PWD}/ce/Paddle_Cloud_CE/src/task/PaddleClas

set +x
rm -rf ce && mkdir ce;
cd ce;

####测试框架下载
"""
通过CE_version判断使用V1还是V2版本
"""
if [[ ${CE_version}=="V1" ]];then
    CE_version_name=${CE_version_name}
    wget -q ${CE_V1}
else
    CE_version_name=continuous_evaluation
    wget -q ${CE_V2}
fi
unzip -P ${CE_pass}  ${CE_version_name}.zip

####设置代理  proxy不单独配置 表示默认有全部配置，不用export
if  [ ! -n "${proxy}" ] ;then
    echo unset http_proxy
else
    export http_proxy=${proxy}
    export https_proxy=${proxy}
fi
export no_proxy=${no_proxy}
set -x;
ls;

####之前下载过了直接mv
mv ../PaddleTest .

set -x;
#通用变量[用户改]
test_code_download_path=./task/models/PaddleClas/CE
test_code_download_path_CI=./task/models/PaddleClas/CI
test_code_conf_path=./task/models/PaddleClas/CE/conf  #各个repo自己管理，可以分类，根据任务类型copy对应的common配置

#迁移下载路径代码和配置到框架指定执行路径 [不用改]
mkdir -p ${test_code_download_path}/log
ls ${test_code_download_path}/log;
cp -r ${test_code_download_path}/.  ./Paddle_Cloud_CE/src/task
cp -r ${test_code_download_path_CI}/.  ./Paddle_Cloud_CE/src/task
cp ${test_code_conf_path}/cls_common_mac.py ./Paddle_Cloud_CE/src/task/common.py
cat ./Paddle_Cloud_CE/src/task/common.py;

#进入执行路径创建docker容器 [用户改docker创建]
cd Paddle_Cloud_CE/src
ls

pwd
echo "begin"
#V2版本
bash main.sh --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type_id=${AGILE_PIPELINE_CONF_ID} --priority=${priority_develop} --compile_path=${compile_path_develop} --job_build_id=${AGILE_JOB_BUILD_ID} 
