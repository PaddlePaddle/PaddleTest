CALL E:\liuquanxiang_env\PaddleClas_py38\Scripts\activate.bat
set PATH=C:\Program Files\Git\bin;C:\Program Files\Git\cmd;C:\Windows\System32;C:\Windows\SysWOW64;C:\zip_unzip;%PATH%

if exist D:\liuquanxiang\Clas_Py38_Win_102_P0_CE_develop ( rmdir D:\liuquanxiang\Clas_Py38_Win_102_P0_CE_develop /S /Q)
md D:\liuquanxiang\Clas_Py38_Win_102_P0_CE_develop
cd D:\liuquanxiang\Clas_Py38_Win_102_P0_CE_develop
d:
chdir

@ echo off
rmdir ce  /S /Q
md ce
cd ce
rem 测试框架下载
rem 通过CE_version判断使用V1还是V2版本
if [[ ${CE_version}=="V1" ]];then
    CE_version_name=${CE_version_name}
    wget -q ${CE_V1}
else
    CE_version_name=continuous_evaluation
    wget -q ${CE_V2}
fi
unzip -P ${CE_pass}  ${CE_version_name}.zip

rem 设置代理  proxy不单独配置 表示默认有全部配置，不用export
if  [ ! -n "${proxy}" ] ;then
    echo unset http_proxy
else
    export http_proxy=${proxy}
    export https_proxy=${proxy}
fi
set no_proxy=${no_proxy}
ls;

rem 之前下载过了直接mv
move ../PaddleTest .

@ echo on
chdir
dir
md .\task\models\PaddleClas\CE\log
xcopy .\task\models\PaddleClas\CE\.  .\Paddle_Cloud_CE\src\task\  /s /e
xcopy .\task\models\PaddleClas\CI\.  .\Paddle_Cloud_CE\src\task\  /s /e
move .\task\models\PaddleClas\CE\conf\cls_common_win.py  .\Paddle_Cloud_CE\src\task\common.py
cd  .\Paddle_Cloud_CE\src
dir;


@ echo off
main.bat --build_id=%AGILE_PIPELINE_BUILD_ID% --build_type_id=%AGILE_PIPELINE_CONF_ID% --priority=%priority_develop% --compile_path=%compile_path_develop% --job_build_id=%AGILE_JOB_BUILD_ID%
