@echo off
cd ../..

set logpath=%cd%\log\taskflow

cd task/taskflow

python interact.py > %logpath%/train_win.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_win.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_win.log
)
