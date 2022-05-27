@echo off
set root_path=%cd%
echo %root_path%

::存放 PaddleSlim repo代码
if exist ./repos rd /s /q repos
mkdir repos && cd repos
set repo_path=%cd%
echo %repo_path%
cd ..

:: log文件统一存储
if exist ./logs rd /s /q logs
mkdir logs && cd logs
set log_path=%cd%
echo %log_path%
cd ..


::下载数据集
if exist ./data rd /s /q data
mkdir data && cd data
wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz --no-check-certificate
tar xf ILSVRC2012_data_demo.tar.gz

wget -q https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/cifar-100-python.tar.gz --no-check-certificate
tar xf cifar-100-python.tar.gz

wget -q https://sys-p0.bj.bcebos.com/slim_ci/word_2evc_demo_data.tar.gz --no-check-certificate
tar xf word_2evc_demo_data.tar.gz

set data_path=%cd%
echo %data_path%
cd ..

::下载预训练模型
set root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
set pre_models="MobileNetV1 MobileNetV2 MobileNetV3_large_x1_0_ssld ResNet101_vd MobileNetV2 ResNet34 ResNet50 ResNet50_vd"
if exist ./pretrain rd /s /q pretrain
mkdir pretrain && cd pretrain

setlocal enabledelayedexpansion
for %%i in (MobileNetV1 MobileNetV2 MobileNetV3_large_x1_0_ssld ResNet101_vd MobileNetV2 ResNet34 ResNet50 ResNet50_vd) do (
echo --------wget %%i---------------
wget -q %root_url%/%%i_pretrained.tar
tar xf %%i_pretrained.tar
)
cd ..
set pretrain_path=%cd%\pretrain
echo ----------dir------
dir

:: compile_path slim_branch
call slim_prepare_env.bat %1 %2

cd %root_path%
if "%3"=="run_P0" (
	echo ----run P0 case ---
	call  slim_run_case_windows_P0.bat
) else if "%3"=="run_P1" (
	echo ----run P1 case ---
	call  slim_run_case_windows_P1.bat
	call :prinf_logs
) else if "%3"=="run_CPU"  (
	echo ----run CPU case ---
	call  slim_run_case_windows_CPU.bat
)else (
    	echo ---skip run case with windows---
)

:prinf_logs
cd %root_path%
cd %log_path%
for /f "delims=" %%i in (' find /C "FAIL" result.log ') do set result=%%i
echo %result:~-1%

for /f "delims=" %%i in (' echo %result:~-1% ') do set exitcode=%%i
echo -----fail case:%exitcode%---------
echo -----exit code:%exitcode%---------
exit %exitcode%
goto :eof

goto :eof
