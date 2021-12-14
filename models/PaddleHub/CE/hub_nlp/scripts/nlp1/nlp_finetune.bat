@echo off

set cur_path=%cd%
echo "++++++++++++++++++++++++++++++++%1 begin to finetune !!!!!!!!!++++++++++++++++++++++++++++++++"

set log_path=%cur_path%

del /q %1_finetune_%2_%3_%4_%5_%6_%7_%8_%9.log
del /q EXIT_%1_finetune_%2_%3_%4_%5_%6_%7_%8_%9.log

setlocal enabledelayedexpansion
python nlp_finetune.py --model_name %1 --task %2 --train_data %3 --use_gpu %4 --batch_size %5 --num_epoch %6 --dataset %7 --optimizer %8 --save_interval %9 >> %log_path%/%1_finetune_%2_%3_%4_%5_%6_%7_%8_%9.log 2>&1

if not !errorlevel! == 0 (
    echo "exit_code: 1.0" >> %log_path%/EXIT_%1_finetune_%2_%3_%4_%5_%6_%7_%8_%9.log
) else (
    echo "exit_code: 0.0" >> %log_path%/EXIT_%1_finetune_%2_%3_%4_%5_%6_%7_%8_%9.log
)
