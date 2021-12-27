@ echo off

set log_path=log
md log
python -m pip install -r requirements.txt
if not exist train_data (mklink /j train_data %data_path%\PaddleOCR\train_data)
if not exist pretrain_models (mklink /j pretrain_models %data_path%\PaddleOCR\pretrain_models)


set gpu_flag=True
set sed="C:\Program Files\Git\usr\bin\sed.exe"
setlocal enabledelayedexpansion
for /f %%i in (ocr_det_models_list.txt) do (
echo %%i
%sed% -i s/"training"/"validation"/g %%i
set target=%%i
rem echo !target!
set target1=!target:*/=!
rem echo !target1!
set target2=!target1:*/=!
rem echo !target2!
set model=!target2:.yml=!
echo !model!
rem train
del %log_path%\result.log
type nul > %log_path%\result.log
python tools/train.py -c %%i  -o Global.use_gpu=True Global.epoch_num=1 Global.save_epoch_step=1 Global.save_model_dir="output/"!model! Train.loader.batch_size_per_card=4 > %log_path%/!model!_train.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,train,FAIL  >> %log_path%\result.log
        echo  training of !model! failed!
) else (
        echo   !model!,train,SUCCESS  >> %log_path%\result.log
        echo   training of !model! successfully!
)

python tools/eval.py -c %%i  -o Global.use_gpu=True Global.checkpoints="output/"!model!"/latest" > %log_path%/!model!_eval.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,eval,FAIL  >> %log_path%\result.log
        echo  evaling of !model! failed!
) else (
        echo   !model!,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of !model! successfully!
)

python tools/infer_det.py -c %%i  -o Global.use_gpu=True Global.checkpoints="output/"!model!"/latest" Global.infer_img="./doc/imgs_en/" Global.test_batch_size_per_card=1 > %log_path%/!model!_infer.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,infer,FAIL  >> %log_path%\result.log
        echo  infering of !model! failed!
) else (
        echo   !model!,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of !model! successfully!
)

python tools/export_model.py -c %%i -o Global.use_gpu=True Global.checkpoints="output/"!model!"/latest"  Global.save_inference_dir="./models_inference/"!model! > %log_path%/!model!_export.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,export_model,FAIL  >> %log_path%\result.log
        echo  export_modeling of !model! failed!
) else (
        echo   !model!,export_model,SUCCESS  >> %log_path%\result.log
        echo   export_model of !model! successfully!
)
python tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./models_inference/"!model! > %log_path%/!model!_predict.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,predict,FAIL  >> %log_path%\result.log
        echo  predicting of !model! failed!
) else (
        echo   !model!,predict,SUCCESS  >> %log_path%\result.log
        echo   predicting of !model! successfully!
)
)


rem *************************
setlocal enabledelayedexpansion
for /f %%i in (ocr_det_models_list_2.txt) do (
echo %%i
set target=%%i
rem echo !target!
set target1=!target:*/=!
rem echo !target1!
set target2=!target1:*/=!
set target2=!target2:*/=!
rem echo !target2!
set model=!target2:.yml=!
echo !model!
python tools/train.py -c %%i  -o Global.use_gpu=True Global.epoch_num=1 Global.save_epoch_step=1 Global.save_model_dir="output/"!model! Train.loader.batch_size_per_card=4 > %log_path%/!model!_train.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,train,FAIL  >> %log_path%\result.log
        echo  training of !model! failed!
) else (
        echo   !model!,train,SUCCESS  >> %log_path%\result.log
        echo   training of !model! successfully!
)

python tools/eval.py -c %%i  -o Global.use_gpu=True Global.checkpoints="output/"!model!"/latest" > %log_path%/!model!_eval.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,eval,FAIL  >> %log_path%\result.log
        echo  evaling of !model! failed!
) else (
        echo   !model!,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of !model! successfully!
)

python tools/infer_det.py -c %%i  -o Global.use_gpu=True Global.checkpoints="output/"!model!"/latest" Global.infer_img="./doc/imgs_en/" Global.test_batch_size_per_card=1 > %log_path%/!model!_infer.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,infer,FAIL  >> %log_path%\result.log
        echo  infering of !model! failed!
) else (
        echo   !model!,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of !model! successfully!
)

python tools/export_model.py -c %%i -o Global.use_gpu=True Global.checkpoints="output/"!model!"/latest"  Global.save_inference_dir="./models_inference/"!model! > %log_path%/!model!_export.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,export_model,FAIL  >> %log_path%\result.log
        echo  export_modeling of !model! failed!
) else (
        echo   !model!,export_model,SUCCESS  >> %log_path%\result.log
        echo   export_model of !model! successfully!
)
python tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./models_inference/"!model! > %log_path%/!model!_predict.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,predict,FAIL  >> %log_path%\result.log
        echo  predicting of !model! failed!
) else (
        echo   !model!,predict,SUCCESS  >> %log_path%\result.log
        echo   predicting of !model! successfully!
)
echo -----------------------------------------------------------
)


rem exit
set num=0
for /F %%i in ('findstr /s "FAIL" %log_path%/result.log') do ( set num=%%i )
findstr /s "FAIL" %log_path%/result.log
if %num%==0 (exit /b 0)else (exit /b 1)
