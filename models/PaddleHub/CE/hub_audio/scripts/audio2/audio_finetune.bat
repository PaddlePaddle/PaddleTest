@echo off

set cur_path=%cd%
echo "++++++++++++++++++++++++++++++++%1 begin to finetune !!!!!!!!!++++++++++++++++++++++++++++++++"

set log_path=%cur_path%

del /q %1_finetune_%2_%3_%4_%5_%6_%7.log
del /q EXIT_%1_finetune_%2_%3_%4_%5_%6_%7.log

setlocal enabledelayedexpansion
python audio_finetune.py --model_name %1 --task %2 --use_gpu %3 --batch_size %4 --num_epoch %5 --learning_rate %6 --save_interval %7 --checkpoint_dir %8 >> %log_path%/%1_finetune_%2_%3_%4_%5_%6_%7.log 2>&1
if not !errorlevel! == 0 (
    echo "exit_code: 1.0" >> %log_path%/EXIT_%1_finetune_%2_%3_%4_%5_%6_%7.log
) else (
    echo "exit_code: 0.0" >> %log_path%/EXIT_%1_finetune_%2_%3_%4_%5_%6_%7.log
)
