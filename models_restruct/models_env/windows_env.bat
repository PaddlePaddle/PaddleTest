
rem do not uset chinese note  beacause of linux tar problem

rem change path
set sed="C:\Program Files\Git\usr\bin\sed.exe"
set pwd_org=%cd%
echo "org path is %pwd_org%"
echo del useless files
for /d /r "D:\Windows_env" %%i in (~*) do rd /s /q "%%i"
if not defined AGILE_PIPELINE_NAME (
    echo "do not exit AGILE_PIPELINE_NAME, use org path %pwd_org%"
    set AGILE_PIPELINE_NAME=test
)  else  (
    if exist D:\PaddleMT\%AGILE_PIPELINE_NAME% ( rmdir D:\PaddleMT\%AGILE_PIPELINE_NAME% /S /Q)
    md D:\PaddleMT\%AGILE_PIPELINE_NAME%
    cd D:\PaddleMT\%AGILE_PIPELINE_NAME%
    d:
    echo "change path to D:\PaddleMT\%AGILE_PIPELINE_NAME%, so do not run same xly task in the same time"
    chdir
)

@ echo off
rmdir ce  /S /Q
rem set root
md ce
cd ce

rem AGILE_PIPELINE_NAME    must set as:   PaddleClas-Windows-Cuda117-Python310-P0-Develop
rem do not use "-",   split as  "-"  so please set value in order

rem reponame
if not defined reponame for /f "tokens=1 delims=-" %%a in ("%AGILE_PIPELINE_NAME%") do set reponame=%%a

rem load self reponame data
rem do not use ""
set mount_path=H:\MT_data\%reponame%
echo mount_path: %mount_path%
dir %mount_path%

rem must set as: tools/reponame_priority_list   do not use "-" , in case mix "-" split
if not defined models_file for /f "tokens=5 delims=-" %%a in ("%AGILE_PIPELINE_NAME%") do set models_file="tools/%reponame%_%%a_list"
if not defined models_list set models_list=None

rem system
rem need first judgment windowscpu
echo %AGILE_PIPELINE_NAME% | findstr "WindowsCPU" >nul
if %errorlevel% equ 0 (
    if not defined system set system=windows_cpu
)  else  (
    if not defined system set system=windows
)
if not defined system set system=windows

rem Python_version
if not defined Python_version for /f "tokens=4 delims=-" %%a in ("%AGILE_PIPELINE_NAME%") do set Python_version=%%a
rem not xly use default Python_version
if not defined Python_version set Python_version=Python310
echo %Python_version% | findstr "37" >nul
if %errorlevel% equ 0 (
    @REM CALL conda activate %reponame%_py37
    @REM C:\Python37\Scripts\virtualenv %reponame%_py37
    CALL D:\Windows_env\%reponame%_py37\Scripts\activate.bat
    %sed% -i s/"include-system-site-packages = false"/"include-system-site-packages = true"/g D:\Windows_env\%reponame%_py37\pyvenv.cfg
    type D:\Windows_env\%reponame%_py37\pyvenv.cfg
)
echo %Python_version% | findstr "38" >nul
if %errorlevel% equ 0 (
    @REM CALL conda activate %reponame%_py38
    @REM C:\Python38\Scripts\virtualenv %reponame%_py38
    CALL D:\Windows_env\%reponame%_py38\Scripts\activate.bat
    %sed% -i s/"include-system-site-packages = false"/"include-system-site-packages = true"/g D:\Windows_env\%reponame%_py38\pyvenv.cfg
    type D:\Windows_env\%reponame%_py38\pyvenv.cfg
)
echo %Python_version% | findstr "39" >nul
if %errorlevel% equ 0 (
    @REM CALL conda activate %reponame%_py39
    @REM C:\Python39\Scripts\virtualenv %reponame%_py39
    CALL D:\Windows_env\%reponame%_py39\Scripts\activate.bat
    %sed% -i s/"include-system-site-packages = false"/"include-system-site-packages = true"/g D:\Windows_env\%reponame%_py39\pyvenv.cfg
    type D:\Windows_env\%reponame%_py39\pyvenv.cfg
)
echo %Python_version% | findstr "310" >nul
if %errorlevel% equ 0 (
    @REM CALL conda activate %reponame%_py310
    @REM C:\Python310\Scripts\virtualenv %reponame%_py310
    CALL D:\Windows_env\%reponame%_py310\Scripts\activate.bat
    %sed% -i s/"include-system-site-packages = false"/"include-system-site-packages = true"/g D:\Windows_env\%reponame%_py310\pyvenv.cfg
    type D:\Windows_env\%reponame%_py310\pyvenv.cfg
)

rem set path
set "PATH=C:\Program Files\Git\bin;C:\Program Files\Git\cmd;C:\Windows\System32;C:\Windows\SysWOW64;C:\zip_unzip;D:\TensorRT-8.4.1.5\lib;%PATH%"

rem cuda_version
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
echo %AGILE_PIPELINE_NAME% | findstr "\Cuda117" >nul
if %errorlevel% equ 0 (
    set cuda_version=11.7
)
rem not xly use default cuda_version
if not defined cuda_version set cuda_version=11.7
set "PATH=C:\Program Files\NVIDIA Corporation\NVSMI;%PATH%"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%cuda_version%\libnvvp;%PATH%"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%cuda_version%\bin;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%cuda_version%"
rem nvcc -V

rem paddle_whl
echo %AGILE_PIPELINE_NAME% | findstr "Develop" >nul
if %errorlevel% equ 0 (
    set flag_branch=1
)  else  (
    set flag_branch=0
)
echo flag_branch %flag_branch%

echo %AGILE_PIPELINE_NAME% | findstr "Intel-Python310" >nul
if %errorlevel% equ 0 (
    set flag_version=310
)  else  (
    set flag_version=0
)
echo flag_version %flag_version%
if %flag_version% equ 310 (
    if %flag_branch% equ 1 (
        if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp310-cp310-win_amd64.whl"
    )  else  (
        if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/release/2.5/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp310-cp310-win_amd64.whl"
    )
)

echo %AGILE_PIPELINE_NAME% | findstr "Cuda117-Python310" >nul
if %errorlevel% equ 0 (
    set flag_version=117310
)  else  (
    set flag_version=0
)
echo flag_version %flag_version%
if %flag_version% equ 117310 (
    if %flag_branch% equ 1 (
        if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-win_amd64.whl"

    )  else  (
        if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/release/2.5/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-win_amd64.whl"
    )
)

echo %AGILE_PIPELINE_NAME% | findstr "Cuda116-Python39" >nul
if %errorlevel% equ 0 (
    set flag_version=11639
)  else  (
    set flag_version=0
)
echo flag_version %flag_version%
if %flag_version% equ 11639 (
    if %flag_branch% equ 1 (
        if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-win_amd64.whl"
    )  else  (
        if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/release/2.5/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-win_amd64.whl"
    )
)

echo %AGILE_PIPELINE_NAME% | findstr "Cuda112-Python38" >nul
if %errorlevel% equ 0 (
    set flag_version=11238
)  else  (
    set flag_version=0
)
echo flag_version %flag_version%
if %flag_version% equ 11238 (
    if %flag_branch% equ 1 (
        if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp38-cp38-win_amd64.whl"
    )  else  (
        if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/release/2.5/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp38-cp38-win_amd64.whl"
    )
)

rem not xly use default paddle_whl
if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-win_amd64.whl"
@REM if not defined paddle_whl set paddle_whl="https://paddle-wheel.bj.bcebos.com/release/2.5/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-win_amd64.whl"

rem default value
if not defined step set step=train:all+eval:all+infer:all+export:all+predict:all
if not defined mode set mode=function
if not defined timeout set timeout=3600
if not defined use_build set use_build=yes
if not defined branch set branch=develop
if not defined get_repo set get_repo=wget
if not defined dataset_org set dataset_org=None
if not defined dataset_target set dataset_target=None

if not defined binary_search_flag set binary_search_flag=False
if not defined use_data_cfs set use_data_cfs=False


rem expend value
if not defined http_proxy set http_proxy=
if not defined no_proxy set no_proxy=

rem maybe change value
if not defined CE_version_name set CE_version_name=TestFrameWork
if not defined models_name set models_name=models_restruct

rem download CE_Link
wget -q %CE_Link%
unzip -P %CE_pass% %CE_version_name%.zip

rem set proxy
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
rem echo value
echo "@@@reponame: %reponame%"
echo "@@@models_list: %models_list%"
echo "@@@models_file: %models_file%"
echo "@@@system: %system%"
echo "@@@Python_version: %Python_version%"
echo "@@@paddle_whl: %paddle_whl%"
echo "@@@step: %step%"
echo "@@@branch: %branch%"
echo "@@@mode: %mode%"
echo "@@@timeout: %timeout%"
echo "@@@dataset_org: %dataset_org%"
echo "@@@dataset_target: %dataset_target%"
echo "@@@binary_search_flag: %binary_search_flag%"
echo "@@@use_data_cfs: %use_data_cfs%"

rem if already download PaddleTest direct mv
if exist "%pwd_org%/task" (
    echo "move org task"
    move %pwd_org%/task .
) else (
    echo "download PaddleTest.tar.gz"
    wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz --no-proxy
    tar xf PaddleTest.tar.gz
    move PaddleTest task
)

rem if already download reponame direct mv
if exist "%pwd_org%/%reponame%" (
    echo D| echo A| XCOPY "%pwd_org%/%reponame%" %reponame%
    echo "because %reponame% already download direct use %reponame%"
)

chdir
dir
rem copy file
xcopy  .\task\%models_name%\%reponame%\. .\%CE_version_name%\ /s /e
cd .\%CE_version_name%\
dir

rem python version
python  --version
git --version
rem org install
py -3.%Python_version:Python3=% -m pip install -U pip -i https://mirror.baidu.com/pypi/simple
py -3.%Python_version:Python3=% -m pip install -U -r requirements.txt -i https://mirror.baidu.com/pypi/simple
python -m pip install -U pip -i https://mirror.baidu.com/pypi/simple
python -m pip install -U -r requirements.txt -i https://mirror.baidu.com/pypi/simple
rem kill python.exe in case can not uninstall sit-package
rem python -c "import os;os.system('taskkill /f /im %s % python.exe')"
rem install package
python main.py --models_list=%models_list% --models_file=%models_file% --system=%system% --step=%step% --reponame=%reponame% --mode=%mode% --use_build=%use_build% --branch=%branch% --get_repo=%get_repo% --paddle_whl=%paddle_whl% --dataset_org=%dataset_org% --dataset_target=%dataset_target% --timeout=%timeout%  --binary_search_flag=%binary_search_flag% --use_data_cfs=%use_data_cfs%
