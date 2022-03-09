@ echo off
@REM set model_flag=CE

echo %1
echo %data_path%
echo %Project_path%
echo %model_flag%

echo "path before"
chdir
setlocal enabledelayedexpansion
echo "CE"| findstr %model_flag% >nul
if !errorlevel! equ 0 (
	echo "CE step"
	set FLAGS_cudnn_deterministic=True
	cd %Project_path%
	echo "path after"
	echo %1 >clas_models_list_P0_cpu
	chdir
	dir
)

set log_path=log
set gpu_flag=True
if exist "log" (
   rmdir log /S /Q
	md log
) else (
	md log
)
cd dataset
if not exist ILSVRC2012 (mklink /j ILSVRC2012 %data_path%\PaddleClas\ILSVRC2012)
cd ..

rem dependency
python -m pip install -r requirements.txt
python -c "import paddle; print(paddle.__version__,paddle.version.commit)"
set sed="C:\Program Files\Git\usr\bin\sed.exe"

for /f %%i in (clas_models_list_P0_cpu) do (
rem echo %%i

set target=%%i
rem echo !target!
set target1=!target:*/=!
set target2=!target1:*/=!
set target2=!target2:*/=!
set target2=!target2:*/=!
set model=!target2:.yaml=!
echo !model!
rem nvidia-smi
rem train

if exist "output" (
   echo "!model! output  exist"
   rmdir output /S /Q
) else (
   echo "!model! output not exist"
)

echo "GoogLeNet VGG11 ViT_small_patch16_224 PPLCNet_x1_0 MobileNetV3_large_x1_0 RedNet50 TNT_small LeViT_128S GhostNet_x1_3"| findstr !model! >nul
if !errorlevel! equ 0 (
	%sed% -i s/"learning_rate:"/"learning_rate: 0.0001 #"/g %%i
	echo "change lr"
)

echo "CE"| findstr %model_flag% >nul
if !errorlevel! equ 0 (
    python tools/train.py -c %%i -o Global.epochs=2 -o DataLoader.Train.sampler.batch_size=1 -o DataLoader.Eval.sampler.batch_size=1 -o Global.output_dir=output -o Global.seed=1234  -o DataLoader.Train.loader.num_workers=0 -o DataLoader.Train.sampler.shuffle=False -o Global.eval_interval=2 -o Global.save_interval=2  -o Global.device=cpu > %log_path%\!model!_train.log 2>&1
) else (
    python tools/train.py -c %%i -o Global.epochs=2 -o DataLoader.Train.sampler.batch_size=1 -o Global.output_dir=output -o DataLoader.Eval.sampler.batch_size=1  -o Global.device=cpu> %log_path%\!model!_train.log 2>&1
)

if not !errorlevel! == 0 (
        type   %log_path%\!model!_train.log
        echo   !model!,train,FAIL  >> %log_path%\result.log
        echo  training of !model! failed!
        echo "training_exit_code: 1.0" >> %log_path%\!model!_train.log
) else (
        echo   !model!,train,SUCCESS  >> %log_path%\result.log
        echo   training of !model! successfully!
        echo "training_exit_code: 0.0" >> %log_path%\!model!_train.log

)

echo "MobileNetV3_large_x1_0 PPLCNet_x1_0 ESNet_x1_0 ResNet50 ResNet50_vd"| findstr !model! >nul
if !errorlevel! equ 0 (
    echo ######  use pretrain model
    echo !model!
    del "output\!model!\latest.pdparams"
    wget -q -P tmp\ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/!model!_pretrained.pdparams --no-proxy --no-check-certificate
    echo f| xcopy /s /y /F "tmp\!model!_pretrained.pdparams" "output\!model!\latest.pdparams"
    rmdir tmp /S /Q
) else (
    echo ######   not load pretrain
)

echo "RedNet50 LeViT_128S GhostNet_x1_3 RedNet50 TNT_small LeViT_128S SwinTransformer_large_patch4_window12_384"| findstr !model! >nul
if !errorlevel! equ 0 (
    echo ######  use pretrain model
    echo !model!
    del "output\!model!\latest.pdparams"
    wget -q -P tmp\ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/!model!_pretrained.pdparams --no-proxy  --no-check-certificate
    echo f| xcopy /s /y /F "tmp\!model!_pretrained.pdparams" "output\!model!\latest.pdparams"
    rmdir tmp /S /Q
) else (
    echo ######   not load pretrain
)

rem eval
python tools/eval.py -c %%i -o Global.pretrained_model="./output/!model!/latest" -o DataLoader.Eval.sampler.batch_size=1 -o Global.device=cpu >%log_path%/!model!_eva.log 2>&1
if not !errorlevel! == 0 (
        type   %log_path%\!model!_eval.log
        echo   !model!,eval,FAIL  >> %log_path%\result.log
        echo  evaling of !model! failed!
        echo "eval_exit_code: 1.0" >> %log_path%\!model!_eval.log
) else (
        echo   !model!,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of !model! successfully!
        echo "eval_exit_code: 0.0" >> %log_path%\!model!_eval.log
)

rem infer
python tools/infer.py -c %%i -o Global.pretrained_model="./output/!model!/latest" -o Global.device=cpu > %log_path%/!model!_infer.log 2>&1
if not !errorlevel! == 0 (
        type   %log_path%\!model!_infer.log
        echo   !model!,infer,FAIL  >> %log_path%\result.log
        echo  infering of !model! failed!
        echo "infer_exit_code: 1.0" >> %log_path%\!model!_infer.log
) else (
        echo   !model!,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of !model! successfully!
        echo "infer_exit_code: 0.0" >> %log_path%\!model!_infer.log
)

rem export_model
python tools/export_model.py -c  %%i -o Global.pretrained_model="./output/!model!/latest" -o Global.save_inference_dir=./inference/!model! -o Global.device=cpu >%log_path%/!model!_export_model.log 2>&1
if not !errorlevel! == 0 (
        type   %log_path%\!model!_export.log
        echo   !model!,export_model,FAIL  >> %log_path%\result.log
        echo  export_modeling of !model! failed!
        echo "export_exit_code: 1.0" >> %log_path%\!model!_export.log
) else (
        echo   !model!,export_model,SUCCESS  >> %log_path%\result.log
        echo   export_model of !model! successfully!
        echo "export_exit_code: 0.0" >> %log_path%\!model!_export.log
)
rem predict
cd deploy
python python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir="../inference/!model!" -o Global.use_gpu=False > ../%log_path%/!model!_predict.log 2>&1
if not !errorlevel! == 0 (
        type   ..\%log_path%\!model!_predict.log
        echo   !model!,predict,FAIL  >> ..\%log_path%\result.log
        echo  predicting of !model! failed!
        echo "predict_exit_code: 1.0" >> ..\%log_path%\!model!_predict.log
) else (
        echo   !model!,predict,SUCCESS  >> ..\%log_path%\result.log
        echo   predicting of !model! successfully!
        echo "predict_exit_code: 0.0" >> ..\%log_path%\!model!_predict.log
)
cd ..
rem TIMEOUT /T 10
)

@REM rmdir dataset /S /Q
rem 清空数据文件防止效率云清空任务时删除原始文件
set num=0
for /F %%i in ('findstr /s "FAIL" %log_path%\result.log') do ( set num=%%i )
findstr /s "FAIL" %log_path%\result.log
rem echo %num%

if %num%==0 (
exit /b 0
) else (
exit /b 1
)
