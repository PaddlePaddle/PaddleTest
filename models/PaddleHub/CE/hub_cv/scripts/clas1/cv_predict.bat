@echo off

set cur_path=%cd%
echo "++++++++++++++++++++++++++++++++%1 begin to predict !!!!!!!!!++++++++++++++++++++++++++++++++"

set log_path=%cur_path%

del /q %1_predict_%2.log
del /q EXIT_%1_predict_%2.log

setlocal enabledelayedexpansion
python cv_predict.py --model_name %1 --use_gpu %2 --img_path %3 >> %log_path%/%1_predict_%2.log 2>&1
if not !errorlevel! == 0 (
    echo "exit_code: 1.0" >> %log_path%/EXIT_%1_predict_%2.log
) else (
    echo "exit_code: 0.0" >> %log_path%/EXIT_%1_predict_%2.log
)
