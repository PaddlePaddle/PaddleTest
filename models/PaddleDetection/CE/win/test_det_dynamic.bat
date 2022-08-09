@echo off
@setlocal enabledelayedexpansion
set CUDA_VISIBLE_DEVICES=0
set sed="C:\Program Files\Git\usr\bin\sed.exe"
rem prepare dynamic data
%sed% -i "s/trainval.txt/test.txt/g" configs/datasets/voc.yml
rem modify coco images
%sed% -i 's/coco.getImgIds()/coco.getImgIds()[:100]/g' ppdet/data/source/coco.py
%sed% -i 's/coco.getImgIds()/coco.getImgIds()[:2]/g' ppdet/data/source/keypoint_coco.py
%sed% -i 's/records, cname2cid/records[:2], cname2cid/g' ppdet/data/source/voc.py
rem modify dynamic_train_iter
%sed% -i '/for step_id, data in enumerate(self.loader):/i\            max_step_id =90' ppdet/engine/trainer.py
%sed% -i '/for step_id, data in enumerate(self.loader):/a\                if step_id == max_step_id: break' ppdet/engine/trainer.py
rem mot_eval iter
%sed% -i '/for seq in seqs/for seq in [seqs[0]]/g' ppdet/engine/tracker.py
%sed% -i '/for step_id, data in enumerate(dataloader):/i\        max_step_id=1' ppdet/engine/tracker.py
%sed% -i '/for step_id, data in enumerate(dataloader):/a\            if step_id == max_step_id: break' ppdet/engine/tracker.py
rem change the pretrained model dir
%sed% -i 's#~/.cache/paddle/weights#D:/ce_data/paddledetection/det_pretrained#g' ppdet/utils/download.py


rem create log dir
if exist log (
rem del /f /s /q log/*.*
rd /s /q log
md log
) else (
md log
)
if exist log_err (
rem del /f /s /q log/*.*
rd /s /q log_err
md log_err
) else (
md log_err
)

rem compile op
cd ppdet/ext_op
python setup.py install
cd ../..

rem start test
set err_sign=0
if exist det_dynamic_list (
del /f det_dynamic_list
)
for /r configs %%i in (*.yml) do (
echo %%i | findstr /i .yml | findstr /v /i "_base_" | findstr /v /i "kunlun" | findstr /v /i "reader" | findstr /v /i "test" | findstr /v /i "oidv5" | findstr /v /i "datasets" | findstr /v /i "runtime" | findstr /v /i "slim" | findstr /v /i "roadsign" | findstr /v /i "minicoco" | findstr /v /i "mot" | findstr /v /i "pruner" | findstr /v /i "pedestrian_detection" | findstr /v /i "keypoint" | findstr /v /i "smrt" | findstr /v /i "xpu" | findstr /v /i "ocsort" | findstr /v /i "pphuman" | findstr /v /i "ppvehicle" >>det_dynamic_list
)
echo test_start !
set absolute_path=%cd%
for  /f %%i in (det_dynamic_list) do (
echo %%i
set config_path=%%i
set config_name=%%~nxi
echo this config is !config_name!
for /f "tokens=1 delims=." %%a in ("!config_name!") do (
set model=%%a
)
cd log
md !model!
cd ..
set infer_method=infer
set eval_method=eval
set python_infer_method=infer
echo %%i | findstr /i "mot"
if !errorlevel! EQU 0 (
    set infer_method=infer_mot
	set eval_method=eval_mot
	set python_infer_method=mot_jde_infer
    set weight_dir=mot/
) else (
    echo %%i | findstr /i "keypoint"
	if !errorlevel! EQU 0 (
    set weight_dir=keypoint/
	set python_infer_method=keypoint_infer
    ) else (
    set weight_dir=
)
)
set url=https://paddledet.bj.bcebos.com/models/!weight_dir!!model!.pdparams
findstr /i /c:"!model!" "model_noupload.txt"
if !errorlevel! EQU 0 (
    echo !model! does not upload bos！
    break
) else (
echo !model! | findstr /i "pedestrian_yolov3_darknet"
if !errorlevel! EQU 0 (
    xcopy %absolute_path%\configs\pedestrian\demo\001.png %absolute_path%\demo
    set infer_img=./demo/001.png
    call:infer
    call:export
    call:python_infer
)
echo !model! | findstr /i "vehicle_yolov3_darknet"
if !errorlevel! EQU 0 (
    xcopy %absolute_path%\configs\vehicle\demo\003.png %absolute_path%\demo
    set infer_img=./demo/003.png
    call:infer
    call:export
    call:python_infer
) else (
    echo !model! | findstr /i "s2anet"
    if !errorlevel! EQU 0 (
    set infer_img=./demo/P0072__1.0__0___0.png
    ) else (
    set infer_img=./demo/000000570688.jpg
    )
    call:train
    call:eval
    call:infer
    call:export
    call:python_infer
echo end
)
)
)

if !err_sign! EQU 1 (
exit /b 1
) else (
exit /b 0
)

:train
findstr /i /c:"!model!" "model_skip_train.txt" >tmp_trian
if !errorlevel! EQU 0 (
echo !model! does not test train for some reason!
) else (
python tools/train.py -c !config_path! -o TrainReader.batch_size=1 epoch=2 >log/!model!/!model!_train.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_train.log log_err\!model!\
echo !model!, train, FAIL
echo !model!,train,Failed >>result 2>&1
set err_sign=1
) else (
echo !model!,train, SUCCESS
echo !model!,train,Passed >>result 2>&1
)
)
goto:eof

:eval
findstr /i /c:"!model!" "skip_eval.txt" >tmp_eval
if !errorlevel! EQU 0 (
echo !model! does not test eval for some reason!
) else (
python tools/!eval_method!.py -c !config_path! -o weights=!url! >log/!model!/!model!_eval.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_eval.log log_err\!model!\
echo !model!, eval, FAIL
echo !model!,eval,Failed >>result 2>&1
set err_sign=1
) else (
echo !model!,eval, SUCCESS
echo !model!,eval,Passed >>result 2>&1
)
)
goto:eof

:infer
findstr /i /c:"!model!" "mot_model.txt" >tmp_mot
if !errorlevel! EQU 0 (
python tools/!infer_method!.py -c !config_path! --video_file=./test_demo.mp4 --output_dir=./infer_output/!model!/ -o weights=!url! --save_videos >log/!model!/!model!_infer.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_infer.log log_err\!model!\
echo !model!, infer, FAIL
echo !model!,predict,Failed >>result 2>&1
set err_sign=1
) else (
echo !model!,infer, SUCCESS
echo !model!,predict,Passed >>result 2>&1
)
) else (
findstr /i /c:"!model!" "skip_infer.txt" >tmp_infer
if !errorlevel! EQU 0 (
echo !model! does not test infer for some reason!
) else (
python tools/!infer_method!.py -c !config_path! --infer_img=!infer_img! --output_dir=./infer_output/!model!/ -o weights=!url! >log/!model!/!model!_infer.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_infer.log log_err\!model!\
echo !model!, infer, FAIL
echo !model!,predict,Failed >>result 2>&1
set err_sign=1
) else (
echo !model!,infer, SUCCESS
echo !model!,predict,Passed >>result 2>&1
)
)
)
goto:eof

:export
findstr /i /c:"!model!" "skip_export.txt" >tmp_export
if !errorlevel! EQU 0 (
echo !model! does not test export for some reason!
) else (
python tools/export_model.py -c !config_path! --output_dir=./inference_model  -o weights=!url! >log/!model!/!model!_export.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_export.log log_err\!model!\
echo !model!, export_model, FAIL
echo !model!,export,Failed >>result 2>&1
set err_sign=1
) else (
echo !model!,export_model, SUCCESS
echo !model!,export,Passed >>result 2>&1
)
)
goto:eof

:python_infer
findstr /i /c:"!model!" "mot_model.txt" >tmp_mot
if !errorlevel! EQU 0 (
set PYTHONPATH=%cd% python deploy/python/!python_infer_method!.py --model_dir=./inference_model/!model! --video_file=./test_demo.mp4 --device=GPU --run_mode=fluid --threshold=0.5 --output_dir=python_infer_output/!model!/ >log/!model!/!model!_python_infer.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_python_infer.log log_err\!model!\
echo !model!, python_infer, FAIL
echo !model!,python_infer,Failed >>result 2>&1
set err_sign=1
) else (
echo !model!,python_infer, SUCCESS
echo !model!,python_infer,Passed >>result 2>&1
)
) else (
findstr /i /c:"!model!" "model_skip_python_infer.txt" >tmp_python_infer
if !errorlevel! GTR 0 (
python deploy/python/!python_infer_method!.py --model_dir=./inference_model/!model! --image_file=!infer_img! --device=GPU --run_mode=fluid --threshold=0.5 --output_dir=python_infer_output/!model!/ --batch_size=1 >log/!model!/!model!_python_infer.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_python_infer.log log_err\!model!\
echo !model!, python_infer, FAIL
echo !model!,python_infer,Failed >>result 2>&1
set err_sign=1
) else (
echo !model!,python_infer, SUCCESS
echo !model!,python_infer,Passed >>result 2>&1
)
) else (
echo !model! does not run python_infer for some reason!
)
)
goto:eof
