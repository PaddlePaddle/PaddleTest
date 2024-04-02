@echo off
cd ../..
if not exist log\rank_%3 md log\rank_%3
set logpath=%cd%\log\rank_%3

cd PaddleRec\models\rank\%3\
echo "%1, %2, %3:" %1, %2, %3
if %1 equ "win_dy_cpu" (
    python -u ..\..\..\tools\infer.py -m config.yaml > %logpath%\%2.log 2>&1
) else if %1 equ "win_st_cpu" (
    python -u ..\..\..\tools\static_infer.py -m config.yaml > %logpath%\%2.log 2>&1
)
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%\%2.log
) else (
    echo "exit_code: 0.0" >> %logpath%\%2.log
)
