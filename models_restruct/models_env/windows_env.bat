@ echo off
rmdir ce  /S /Q
rem 定义根目录
md ce
cd ce

rem ###################### 定义变量 ########################
rem # AGILE_PIPELINE_NAME 格式类似: PaddleClas-Windows-Cuda116-Python310-P0-Develop
rem #其它内容或者可能不一致的不要随意加 "-", 下面是按照 "-" split 按序号填入的

set PATH=C:\Program Files\Git\bin;C:\Program Files\Git\cmd;C:\Windows\System32;C:\Windows\SysWOW64;C:\zip_unzip;%PATH%

rem repo的名称
if not defined reponame for /f "tokens=1 delims=-" %%a in ("%AGILE_PIPELINE_NAME%") do set reponame=%%a

rem #模型列表文件 , 固定路径及格式为 tools/reponame_优先级_list   优先级P2有多个用P21、P22  中间不用"-"划分, 防止按 "-" split 混淆
if not defined models_file for /f "tokens=5 delims=-" %%a in ("%AGILE_PIPELINE_NAME%") do set models_file="tools/%reponame%_%%a_list"
if not defined models_list set models_list=None

rem #指定case操作系统

echo %AGILE_PIPELINE_NAME% | findstr "Windows" >nul
if %errorlevel% equ 0 (
    if not defined system set system=windows
)
echo %AGILE_PIPELINE_NAME% | findstr "WindowsCPU" >nul
if %errorlevel% equ 0 (
    if not defined system set system=windows_cpu
)
if not defined system set system=windows

rem #指定python版本
if not defined Python_version for /f "tokens=4 delims=-" %%a in ("%AGILE_PIPELINE_NAME%") do set Python_version=%%a
rem 如果非流水线设置默认python
if not defined Python_version set Python_version=310
echo %Python_version% | findstr "36" >nul
if %errorlevel% equ 0 (
    CALL D:\Windows_env\%reponame%_py36\Scripts\activate.bat
)
echo %Python_version% | findstr "37" >nul
if %errorlevel% equ 0 (
    CALL D:\Windows_env\%reponame%_py37\Scripts\activate.bat
)
echo %Python_version% | findstr "38" >nul
if %errorlevel% equ 0 (
    CALL D:\Windows_env\%reponame%_py38\Scripts\activate.bat
)
echo %Python_version% | findstr "39" >nul
if %errorlevel% equ 0 (
    CALL D:\Windows_env\%reponame%_py39\Scripts\activate.bat
)
echo %Python_version% | findstr "310" >nul
if %errorlevel% equ 0 (
    CALL D:\Windows_env\%reponame%_py310\Scripts\activate.bat
)

rem #指定python版本
echo %AGILE_PIPELINE_NAME% | findstr "Cuda102" >nul
if %errorlevel% equ 0 (
    set cuda_version=10.2
)
echo %AGILE_PIPELINE_NAME% | findstr "Cuda112" >nul
if %errorlevel% equ 0 (
    set cuda_version=11.2
)
echo %AGILE_PIPELINE_NAME% | findstr "Cuda116" >nul
if %errorlevel% equ 0 (
    set cuda_version=11.6
)
echo %AGILE_PIPELINE_NAME% | findstr "Cuda117" >nul
if %errorlevel% equ 0 (
    set cuda_version=11.7
)
rem 如果非流水线设置默认 cuda_version
if not defined cuda_version set cuda_version=11.7
set "PATH=C:\Program Files\NVIDIA Corporation\NVSMI;%PATH%"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%cuda_version%\libnvvp;%PATH%"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%cuda_version%\bin;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%cuda_version%"
rem nvcc -V

rem #约定覆盖的几条流水线
rem #指定whl包, 暂时用night develop的包
echo %AGILE_PIPELINE_NAME% | findstr "Cuda117" >nul
if %errorlevel% equ 0 (
    echo %Python_version% | findstr "310" >nul
    if %errorlevel% equ 0 (
        echo %AGILE_PIPELINE_NAME% | findstr "Develop" >nul
        if %errorlevel% equ 0 (
            if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-win_amd64.whl"
        )  else  (
            rem Release
            if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/release/2.4/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-win_amd64.whl"
        )
    )
)
echo %AGILE_PIPELINE_NAME% | findstr "Cuda116" >nul
if %errorlevel% equ 0 (
    echo %Python_version% | findstr "39" >nul
    if %errorlevel% equ 0 (
        echo %AGILE_PIPELINE_NAME% | findstr "Develop" >nul
        if %errorlevel% equ 0 (
            if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-win_amd64.whl"
        )  else  (
            rem Release
            if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/release/2.4/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-win_amd64.whl"
        )
    )
)
echo %AGILE_PIPELINE_NAME% | findstr "Cuda112" >nul
if %errorlevel% equ 0 (
    echo %Python_version% | findstr "38" >nul
    if %errorlevel% equ 0 (
        echo %AGILE_PIPELINE_NAME% | findstr "Develop" >nul
        if %errorlevel% equ 0 (
            if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp38-cp38-win_amd64.whl"
        )  else  (
            rem Release
            if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/release/2.4/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp38-cp38-win_amd64.whl"
        )
    )
)
rem 如果非流水线设置默认 paddle_whl
if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-win_amd64.whl"

rem 预设默认参数
if not defined step set step=train
if not defined mode set mode=function
if not defined use_build set use_build=yes
if not defined branch set branch=develop
if not defined get_repo set get_repo=wget
if not defined dataset_org set dataset_org=None
if not defined dataset_target set dataset_target=None

rem 额外的变量
if not defined http_proxy set http_proxy=
if not defined no_proxy set no_proxy=

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
set AK=%AK%
set SK=%SK%
set bce_whl_url=%bce_whl_url%
dir

@ echo on
rem #输出参数验证
echo "@@@reponame: %reponame%"
echo "@@@models_list: %models_list%"
echo "@@@models_file: %models_file%"
echo "@@@system: %system%"
echo "@@@Python_version: %Python_version%"
echo "@@@paddle_whl: %paddle_whl%"
echo "@@@step: %step%"
echo "@@@branch: %branch%"
echo "@@@mode: %mode%"
echo "@@@docker_flag: %docker_flag%"

rem 之前下载过了直接mv
if exist "../task" (
    move ../task .
) else (
    wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz --no-proxy
    tar xf PaddleTest.tar.gz
    move PaddleTest task
)

rem ##如果预先模型库下载直接mv, 方便二分是checkout 到某个commit进行二分
if exist "../../%reponame%" (
    echo D| echo A| XCOPY "../../%reponame%" %reponame%
    echo "因为 %reponame% 在根目录存在 使用预先clone或wget的 %reponame%"
)

chdir
dir
rem 复制模型相关文件到指定位置
xcopy  .\task\%models_name%\%reponame%\. .\%CE_version_name%\ /s /e
cd .\%CE_version_name%\
dir

rem 查看python版本
python  --version
git --version
python -m pip install -U pip
python -m pip install -r requirements.txt
rem 预先安装依赖包
python main.py --models_list=%models_list% --models_file=%models_file% --system=%system% --step=%step% --reponame=%reponame% --mode=%mode% --use_build=%use_build% --branch=%branch% --get_repo=%get_repo% --paddle_whl=%paddle_whl% --dataset_org=%dataset_org% --dataset_target=%dataset_target%
