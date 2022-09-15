set +x
rm -rf ce && mkdir ce; #设置根目录
cd ce;

#### 预设参数
Repo=${Repo:-${Repo}}
Python_version=${Python_version:-39}
CE_version=${CE_version:-V1}
Priority_version=${Priority_version:-P0}
Compile_version=${Compile_version:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-Cpu-Mac-Avx-Openblas-Python39-Compile/latest/paddlepaddle-0.0.0-cp39-cp39-macosx_10_14_x86_64.whl}
Data_path=${Data_path:-/Users/paddle/ce_data/PaddleClas}
Common_name=${Common_name:-cls_common_mac}  #CE框架中的执行步骤，名称各异所以需要传入
model_flag=${model_flag:-CE}  #clas gan特有，待完善后删除，可以不设置

#### 激活环境
source ~/.bashrc
pyenv activate ${Repo}_${Python_version}
which python

####查看python版本
python  --version
git --version

####测试框架下载
if [[ ${CE_version} == "V2" ]];then
    CE_version_name=continuous_evaluation
    wget -q ${CE_V2}
else
    CE_version_name=Paddle_Cloud_CE
    wget -q ${CE_V1}
fi
unzip -P ${CE_pass}  ${CE_version_name}.zip

##### 预设Project_path路径
if  [ ! -n "${Project_path}" ] ;then  #通过判断CE框架版本找到绝对路径
    export Project_path=${PWD}/ce/${CE_version_name}/src/task/${Repo}
else
    echo already have Project_path
fi

####设置代理  proxy不单独配置 表示默认有全部配置，不用export
if  [[ ! -n "${http_proxy}" ]] ;then
    echo unset http_proxy
else
    export http_proxy=${http_proxy}
    export https_proxy=${http_proxy}
fi
export no_proxy=${no_proxy}
set -x;
ls;

####之前下载过了直接mv
if [[ -d "../task" ]];then
    mv ../task .  #task路径是CE框架写死的
else
    wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz --no-proxy  >/dev/null
    tar xf PaddleTest.tar.gz >/dev/null 2>&1
    mv PaddleTest task
fi

set -x;
#通用变量[用户改]
test_code_download_path=./task/models/${Repo}/CE
test_code_download_path_CI=./task/models/${Repo}/CI
test_code_conf_path=./task/models/${Repo}/CE/conf  #各个repo自己管理，可以分类，根据任务类型copy对应的common配置

#迁移下载路径代码和配置到框架指定执行路径 [不用改]
mkdir -p ${test_code_download_path}/log
ls ${test_code_download_path}/log;
cp -r ${test_code_download_path}/.  ./${CE_version_name}/src/task
cp -r ${test_code_download_path_CI}/.  ./${CE_version_name}/src/task
cp ${test_code_conf_path}/${Common_name}.py ./${CE_version_name}/src/task/common.py
cat ./${CE_version_name}/src/task/common.py;

#进入执行路径创建docker容器 [用户改docker创建]
cd ${CE_version_name}/src
ls

pwd
echo "begin"
if [[ ${CE_version} == 'V2' ]];then
    #V2版本
    bash main.sh --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type_id=${AGILE_PIPELINE_CONF_ID} --priority=${priority_develop} --compile_path=${compile_path_develop} --job_build_id=${AGILE_JOB_BUILD_ID}
else
    #V1版本
    bash main.sh --task_type='model' --build_number=${AGILE_PIPELINE_BUILD_NUMBER} --project_name=${AGILE_MODULE_NAME} --task_name=${AGILE_PIPELINE_NAME}  --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type=${AGILE_PIPELINE_UUID} --owner='paddle' --priority=${Priority_version} --compile_path=${Compile_version} --agile_job_build_id=${AGILE_JOB_BUILD_ID}
