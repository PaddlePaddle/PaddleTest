@echo off
@setlocal enabledelayedexpansion
set CUDA_VISIBLE_DEVICES=''
set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -i '/samples in file/i\        records = records[:30]' ppdet/data/source/coco.py
%sed% -i 's#~/.cache/paddle/weights#D:/ce_data/paddledetection/det_pretrained#g' ppdet/utils/download.py
%sed% -i '/for step_id, data in enumerate(self.loader):/i\            max_step_id =20' ppdet/engine/trainer.py
%sed% -i '/for step_id, data in enumerate(self.loader):/a\                if step_id == max_step_id: break' ppdet/engine/trainer.py
%sed% -i 's/self.img_ids = self.coco.getImgIds()/self.img_ids = self.coco.getImgIds()[:20]/g' ppdet/data/source/keypoint_coco.py
%sed% -i '/def _load_coco_keypoint_annotations(self):/i\        self.db = self.db[:20]' ppdet/data/source/keypoint_coco.py

rem create log dir
if exist log (
rd /s /q log
md log
) else (
md log
)
if exist log_err (
rd /s /q log_err
md log_err
) else (
md log_err
)

echo test_start !
set absolute_path=%cd%
for  /f %%i in (det_win_cpu_list) do (
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
    echo !model! does not upload
    break
) else (
echo !model! | findstr /i "pedestrian"
if !errorlevel! EQU 0 (
    xcopy %absolute_path%\configs\pedestrian\demo\001.png %absolute_path%\demo
    set infer_img=./demo/001.png
    call:infer
    call:export
    call:python_infer
) else (
    echo !model! | findstr /i "vehicle"
	if !errorlevel! EQU 0 (
    xcopy %absolute_path%\configs\vehicle\demo\003.png %absolute_path%\demo
    set infer_img=./demo/003.png
    call:infer
    call:export
    call:python_infer
    ) else (
    set infer_img=./demo/000000570688.jpg
    call:train
    call:eval
    call:infer
    call:export
    call:python_infer
)
)
)
)

:train
python tools/train.py -c !config_path! -o TrainReader.batch_size=1 epoch=2 use_gpu=false >log/!model!/!model!_train.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_train.log log_err\!model!\
echo !model!, train, FAIL
) else (
echo !model!,train, SUCCESS
)
goto:eof

:eval
findstr /i /c:"!model!" "skip_eval.txt" >tmp_eval
if !errorlevel! EQU 0 (
echo !model! does not test eval for some reason!
) else (
python tools/!eval_method!.py -c !config_path! -o weights=!url! use_gpu=false >log/!model!/!model!_eval.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_eval.log log_err\!model!\
echo !model!, eval, FAIL
) else (
echo !model!,eval, SUCCESS
)
)
goto:eof

:infer
findstr /i /c:"!model!" "mot_model.txt" >tmp_mot
if !errorlevel! EQU 0 (
python tools/!infer_method!.py -c !config_path! --video_file=./test_demo.mp4 --output_dir=./infer_output/!model!/ -o weights=!url! use_gpu=false --save_videos >log/!model!/!model!_infer.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_infer.log log_err\!model!\
echo !model!, infer, FAIL
) else (
echo !model!,infer, SUCCESS
)
) else (
python tools/!infer_method!.py -c !config_path! --infer_img=!infer_img! --output_dir=./infer_output/!model!/ -o weights=!url! use_gpu=false >log/!model!/!model!_infer.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_infer.log log_err\!model!\
echo !model!, infer, FAIL
) else (
echo !model!,infer, SUCCESS
)
)
goto:eof

:export
findstr /i /c:"!model!" "skip_export.txt" >tmp_export
rem echo !model! | findstr /i "cascade" >tmp_export
if !errorlevel! EQU 0 (
echo !model! does not test export for some reason!
) else (
python tools/export_model.py -c !config_path! --output_dir=./inference_model  -o weights=!url! use_gpu=false >log/!model!/!model!_export.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_export.log log_err\!model!\
echo !model!, export_model, FAIL
) else (
echo !model!,export_model, SUCCESS
)
)
goto:eof

:python_infer
findstr /i /c:"!model!" "mot_model.txt" >tmp_mot
if !errorlevel! EQU 0 (
python deploy/python/!python_infer_method!.py --model_dir=./inference_model/!model! --video_file=./test_demo.mp4 --device=CPU  --output_dir=python_infer_output/!model!/ >log/!model!/!model!_python_infer.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_python_infer.log log_err\!model!\
echo !model!, python_infer, FAIL
) else (
echo !model!,python_infer, SUCCESS
)
) else (
python deploy/python/!python_infer_method!.py --model_dir=./inference_model/!model! --image_file=!infer_img! --device=CPU --output_dir=python_infer_output/!model!/ >log/!model!/!model!_python_infer.log 2>&1
if !errorlevel! GTR 0 (
cd log_err && md !model!
cd .. && move log\!model!\!model!_python_infer.log log_err\!model!\
echo !model!, python_infer, FAIL
) else (
echo !model!,python_infer, SUCCESS
)
)
goto:eof
