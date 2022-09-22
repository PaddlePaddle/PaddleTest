@ echo off
rmdir ce  /S /Q
rem 定义根目录
md ce
cd ce

rem 预设默认参数
if not defined models_list set models_list=None
if not defined models_file set models_file=None
if not defined system set system=linux
if not defined step set step=train
if not defined reponame set reponame=PaddleClas
if not defined mode set mode=function
if not defined use_build set use_build=yes
if not defined branch set branch=develop
if not defined get_repo set get_repo=wget
if not defined paddle_whl set paddle_whl=None
if not defined dataset_org set dataset_org=None
if not defined dataset_target set dataset_target=None
if not defined set_cuda set set_cuda=

rem 额外的变量
if not defined docker_flag set docker_flag=
if not defined http_proxy set http_proxy=
if not defined no_proxy set no_proxy=
if not defined Python_env set Python_env=path_way
if not defined Python_version set Python_version=37
if not defined Image_version set Image_version=registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7

rem 预设一些可能会修改的变量
if not defined CE_version_name set CE_version_name=TestFrameWork
if not defined models_name set models_name=models_restruct

rem 测试框架下载
wget -q %CE_Link%
unzip -P %CE_pass% %CE_version_name%.zip

rem 设置代理  proxy不单独配置 表示默认有全部配置，不用export
if not defined http_proxy (
    echo "unset http_proxy"
) else (
    set http_proxy=%http_proxy%
    set https_proxy=%http_proxy%
)
set no_proxy=%no_proxy%
dir

rem 之前下载过了直接mv
if exist "../task" (
    move ../task .
) else (
    wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz --no-proxy
    tar xf PaddleTest.tar.gz
    move PaddleTest task
)

@ echo on
chdir
dir
rem 复制模型相关文件到指定位置
xcopy  .\task\%models_name%\%reponame%\. .\%CE_version_name%\ /s /e
cd .\%CE_version_name%\
dir

rem 查看python版本
python  --version
git --version
python -m pip install -r requirements.txt
rem 预先安装依赖包
python main.py --models_list=%models_list% --models_file=%models_file% --system=%system% --step=%step% --reponame=%reponame% --mode=%mode% --use_build=%use_build% --branch=%branch% --get_repo=%get_repo% --paddle_whl=%paddle_whl% --dataset_org=%dataset_org% --dataset_target=%dataset_target%
