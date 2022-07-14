@ echo off
rmdir ce  /S /Q
rem 定义根目录
md ce
cd ce

rem 预设参数
if not defined Repo set Repo=PaddleClas
if not defined Python_version set Python_version=38
if not defined CE_version set CE_version=V1
if not defined Priority_version set Priority_version=P0
if not defined Compile_version set Compile_version=https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-Avx-Win-Mkl-Cuda102-Py38-Compile/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-win_amd64.whl
if not defined Data_path set Data_path=D:\ce_data
if not defined Project_path set Project_path=D:\liuquanxiang\%Repo%_%Python_version%_Win_102_%Priority_version%_CE_develop\ce\%CE_version_name%\src\task\PaddleClas
if not defined Common_name set Common_name=cls_common_win
if not defined model_flag set model_flag=CE

rem  激活环境，设置环境变量
CALL E:\liuquanxiang_env\%Repo%_%Python_version%\Scripts\activate.bat
set PATH=C:\Program Files\Git\bin;C:\Program Files\Git\cmd;C:\Windows\System32;C:\Windows\SysWOW64;C:\zip_unzip;%PATH%

rem 查看python版本
python  --version
git --version

if exist D:\liuquanxiang\%Repo%_%Python_version%_Win_102_%Priority_version%_CE_develop ( rmdir D:\liuquanxiang\%Repo%_%Python_version%_Win_102_%Priority_version%_CE_develop /S /Q)
md D:\liuquanxiang\%Repo%_%Python_version%_Win_102_%Priority_version%_CE_develop
cd D:\liuquanxiang\%Repo%_%Python_version%_Win_102_%Priority_version%_CE_develop
d:
chdir

rem 测试框架下载
rem 通过 CE_version 判断使用V1还是V2版本

echo "V2"| findstr %CE_version% >nul
if !errorlevel! equ 0 (
    CE_version_name=continuous_evaluation
    wget -q %CE_V2%
) else (
    CE_version_name=Paddle_Cloud_CE
    wget -q %CE_V1%
)
unzip -P %CE_pass%  %CE_version_name%.zip

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
md .\task\models\%Repo%\CE\log
xcopy .\task\models\%Repo%\CE\.  .\%CE_version_name%\src\task\  /s /e
xcopy .\task\models\%Repo%\CI\.  .\%CE_version_name%\src\task\  /s /e
move .\task\models\%Repo%\CE\conf\%Common_name%.py  .\%CE_version_name%\src\task\common.py
cd  .\%CE_version_name%\src
dir;


@ echo off
echo "V2"| findstr %CE_version% >nul
if !errorlevel! equ 0 (
    main.bat --build_id=%AGILE_PIPELINE_BUILD_ID% --build_type_id=%AGILE_PIPELINE_CONF_ID% --priority=%priority% --compile_path=%compile_path_release% --job_build_id=%AGILE_JOB_BUILD_ID%
) else (
    main.bat --task_type='model' --build_number=%AGILE_PIPELINE_BUILD_NUMBER% --project_name=%AGILE_MODULE_NAME% --task_name=%AGILE_PIPELINE_NAME%  --build_id=%AGILE_PIPELINE_BUILD_ID% --build_type=%AGILE_PIPELINE_UUID% --owner='paddle' --priority=%Priority_version% --compile_path=%Compile_version% --agile_job_build_id=%AGILE_JOB_BUILD_ID%
)
