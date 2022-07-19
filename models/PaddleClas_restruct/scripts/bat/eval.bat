@REM # 输入变量：yaml、设置卡数CPU/SET_CUDA/SET_MULTI_CUDA

set sed="C:\Program Files\Git\usr\bin\sed.exe"
setlocal enabledelayedexpansion
CALL prepare.bat %1 %2

cd %Project_path%
@REM  #确定下执行路径


@REM #对训练不足的模型下载预训练模型
echo "MobileNetV3_large_x1_0 PPLCNet_x1_0 ESNet_x1_0 ResNet50 ResNet50_vd"| findstr !model_latest! >nul
if !errorlevel! equ 0 (
    echo ######  use pretrain model
    echo !model!
    del "%output_dir%\!model!\latest.pdparams"
    wget -q -P tmp\ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/!model!_pretrained.pdparams --no-proxy --no-check-certificate
    echo f| xcopy /s /y /F "tmp\!model!_pretrained.pdparams" "%output_dir%\!model!\latest.pdparams"
    rmdir tmp /S /Q
) else (
    echo ######   not load pretrain
)

echo "RedNet50 LeViT_128S GhostNet_x1_3 RedNet50 TNT_small LeViT_128S SwinTransformer_large_patch4_window12_384"| findstr !model_latest! >nul
if !errorlevel! equ 0 (
    echo ######  use pretrain model
    echo !model!
    del "%output_dir%\!model!\latest.pdparams"
    wget -q -P tmp\ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/!model!_pretrained.pdparams --no-proxy  --no-check-certificate
    echo f| xcopy /s /y /F "tmp\!model!_pretrained.pdparams" "%output_dir%\!model!\latest.pdparams"
    rmdir tmp /S /Q
) else (
    echo ######   not load pretrain
)


@REM # 廷权临时增加规则 220413
if %1 == ultra (
    copy %1 %1_tmp
	%sed% -i /output_fp16: True/d %1
)

@REM # 判断是单独评估还是训练后评估
if exist %output_dir%/%model_name% (
    %output_dir%/%model_name%
    set pretrained_model=%output_dir%/%model_name%/%model_latest%/latest
) else (
    set pretrained_model=null
)

rem eval
python tools/eval.py -c %%i -o Global.pretrained_model=!pretrained_model! -o DataLoader.Eval.sampler.batch_size=1 -o Global.device=%set_cuda_device% > %log_path%\!model_name!_eval.log 2>&1
if not !errorlevel! == 0 (
        type   "%log_path%\!model!_eval.log"
        echo   !model!,eval,FAIL  >> %log_path%\result.log
        echo  evaling of !model! failed!
        echo "eval_exit_code: 1.0" >> %log_path%\!model!_eval.log
) else (
        echo   !model!,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of !model! successfully!
        echo "eval_exit_code: 0.0" >> %log_path%\!model!_eval.log
)

@REM # 廷权临时增加规则 220413
if %1 == ultra (
    del %1
    move %1_tmp %1
)
