@echo off

set cur_path=%cd%
echo "++++++++++++++++++++++++++++++++%1 begin to predict !!!!!!!!!++++++++++++++++++++++++++++++++"

set log_path=%cur_path%

del /q %1_predict_%2_%3_%4.log
del /q EXIT_%1_predict_%2_%3_%4.log

setlocal enabledelayedexpansion
python audio_predict.py --model_name %1 --task %2 --use_finetune_model %3 --use_gpu %4 --checkpoint_dir %5 --audio_path %6 >> %log_path%/%1_predict_%2_%3_%4.log 2>&1
if not !errorlevel! == 0 (
    echo "exit_code: 1.0" >> %log_path%/EXIT_%1_predict_%2_%3_%4.log
) else (
    echo "exit_code: 0.0" >> %log_path%/EXIT_%1_predict_%2_%3_%4.log
)
