@echo off
@setlocal enabledelayedexpansion
set CUDA_VISIBLE_DEVICES=''

rem create log dir
if exist log (
rem del /f /s /q log/*.*
rd /s /q log
rd /s /q log_err
md log
) else (
md log
)
md log_err
) else (
md log_err
)

echo test_start
set err_sign=0
for  /f %%i in (dynamic_cpu_list) do (
echo %%i
set config_path=%%i
set config_name=%%~nxi
echo this config is !config_name!
for /f "tokens=1 delims=." %%a in ("!config_name!") do (
set model=%%a
)
echo !model!
cd log
md !model!
cd ..
echo !model! | findstr /i "cityscapes"
if !errorlevel! GTR 0 (
    set predict_pic="2007_000033.jpg"
    if not exist seg_pretrained\!model!\model.pdparams (
        wget -P seg_pretrained\!model! https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/!model!/model.pdparams --no-check-certificate
		)
) else (
    set predict_pic="leverkusen_000029_000019_leftImg8bit.png"
    if not exist seg_pretrained\!model!\model.pdparams (
        wget -P seg_pretrained\!model! https://bj.bcebos.com/paddleseg/dygraph/cityscapes/!model!/model.pdparams --no-check-certificate
    )
)
if not exist seg_pretrained\!model!\model.pdparams (
    echo !model! does not in bos!
) else (
call:train
call:eval
call:predict
call:export
call:python_infer
)
)
if !err_sign! EQU 1 (
exit /b 1
)




:train
findstr /i /c:"!model!" "skip_train.txt" >tmp_train
if !errorlevel! EQU 0 (
echo !model! does not test train for some reason!
) else (
python train.py --config !config_path! --save_interval 100 --iters 10 --save_dir output/!model! --batch_size=1 >log/!model!/!model!_train.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_train.log log_err\!model!\
echo !model!, train, FAIL
set err_sign=1
) else (
echo !model!, train, SUCCESS
)
)
goto:eof

:eval
findstr /i /c:"!model!" "skip_eval.txt" >tmp_eval
if !errorlevel! EQU 0 (
echo !model! does not test eval for some reason!
) else (
python val.py --config !config_path! --model_path seg_pretrained/!model!/model.pdparams >log/!model!/!model!_eval.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_eval.log log_err\!model!\
echo !model!, eval, FAIL
set err_sign=1
) else (
echo !model!, eval, SUCCESS
)
)
goto:eof

:predict
findstr /i /c:"!model!" "skip_predict.txt" >tmp_predict
if !errorlevel! EQU 0 (
echo !model! does not test predict for some reason!
) else (
python predict.py --config !config_path!  --model_path seg_pretrained/!model!/model.pdparams --image_path demo/!predict_pic! --save_dir output/!model!/result >log/!model!/!model!_predict.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_predict.log log_err\!model!\
echo !model!, predict, FAIL
set err_sign=1
) else (
echo !model!, predict, SUCCESS
)
)
goto:eof

:export
findstr /i /c:"!model!" "skip_export.txt" >tmp_export
if !errorlevel! EQU 0 (
echo !model! does not test export for some reason!
) else (
python export.py --config !config_path! --model_path seg_pretrained/!model!/model.pdparams --save_dir ./inference_model/!model! >log/!model!/!model!_export.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_export.log log_err\!model!\
echo !model!, export, FAIL
set err_sign=1
) else (
echo !model!, export, SUCCESS
)
)
goto:eof

:python_infer
findstr /i /c:"!model!" "skip_python_infer.txt" >tmp1_python_infer
if !errorlevel! EQU 0 (
echo !model! does not test python_infer for some reason!
) else (
python deploy/python/infer.py --config ./inference_model/!model!/deploy.yaml --image_path ./demo/!predict_pic! --save_dir ./python_infer_output/!model! >log/!model!/!model!_python_infer.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_python_infer.log log_err\!model!\
echo !model!, python_infer, FAIL
set err_sign=1
) else (
echo !model!, python_infer, SUCCESS
)
)
goto:eof
