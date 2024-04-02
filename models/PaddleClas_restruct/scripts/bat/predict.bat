@REM # 输入变量：yaml、设置卡数CPU/SET_CUDA/SET_MULTI_CUDA

set sed="C:\Program Files\Git\usr\bin\sed.exe"
setlocal enabledelayedexpansion
CALL prepare.bat %1 %2

cd %Project_path%
@REM  #确定下执行路径

cd deploy

@REM # 判断是否有已产出的模型
if exist ../inference/%model_name%/inference.pdparams (
    %output_dir%/%model_name%
    set pretrained_model=%output_dir%/%model_name%/%model_latest%/latest
) else (
    set pretrained_model=null
)

python python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=%pretrained_model% -o Global.use_gpu=%set_cuda_flag% > ../%log_path%/!model_name!_predict.log 2>&1
if not !errorlevel! == 0 (
        type   "..\%log_path%\!model!_predict.log"
        echo   !model!,predict,FAIL  >> ..\%log_path%\result.log
        echo  predicting of !model! failed!
        echo "predict_exit_code: 1.0" >> ..\%log_path%\!model!_predict.log
) else (
        echo   !model!,predict,SUCCESS  >> ..\%log_path%\result.log
        echo   predicting of !model! successfully!
        echo "predict_exit_code: 0.0" >> ..\%log_path%\!model!_predict.log
)
cd ..
