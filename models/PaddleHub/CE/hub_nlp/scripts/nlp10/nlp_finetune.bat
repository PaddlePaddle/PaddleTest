@echo off

set cur_path=%cd%
echo "++++++++++++++++++++++++++++++++%1 begin to finetune !!!!!!!!!++++++++++++++++++++++++++++++++"

set log_path=%cur_path%

del /q %1_finetune_%2_%3_%4_%5_%6.log
del /q EXIT_%1_finetune_%2_%3_%4_%5_%6.log

setlocal enabledelayedexpansion
python nlp_finetune.py --model_name %1 --use_gpu %2 --max_steps %3 --batch_size %4 --module_name %5 --author %6 >> %log_path%/%1_finetune_%2_%3_%4_%5_%6.log 2>&1

if not !errorlevel! == 0 (
    echo "exit_code: 1.0" >> %log_path%/EXIT_%1_finetune_%2_%3_%4_%5_%6.log
) else (
    echo "exit_code: 0.0" >> %log_path%/EXIT_%1_finetune_%2_%3_%4_%5_%6.log
)
