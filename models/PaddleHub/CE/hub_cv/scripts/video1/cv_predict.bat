@echo off

set cur_path=%cd%
echo "++++++++++++++++++++++++++++++++%1 begin to predict !!!!!!!!!++++++++++++++++++++++++++++++++"

set log_path=%cur_path%

del /q %1_predict_%2_%3_%4.log
del /q EXIT_%1_predict_%2_%3_%4.log

setlocal enabledelayedexpansion
python cv_predict.py --model_name %1 --use_gpu %2 --visualization %3 --draw_threshold %4 --video_path %5 --tracking_output_dir %6 --stream_output_dir %7 >> %log_path%/%1_predict_%2_%3_%4.log 2>&1
if not !errorlevel! == 0 (
    echo "exit_code: 1.0" >> %log_path%/EXIT_%1_predict_%2_%3_%4.log
) else (
    echo "exit_code: 0.0" >> %log_path%/EXIT_%1_predict_%2_%3_%4.log
)
