@echo off
cd ../..

if not exist log\taskflow md log\taskflow

set logpath=%cd%\log\taskflow

cd scripts/taskflow

python interact.py > %logpath%/train_win.log 2>&1
