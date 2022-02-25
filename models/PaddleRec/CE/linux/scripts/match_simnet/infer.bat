@echo off
cd ../..
if not exist log\match_simnet md log\match_simnet
set logpath=%cd%\log\match_simnet

cd PaddleRec\models\match\%3\
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
