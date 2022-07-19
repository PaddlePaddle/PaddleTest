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
    del "output\!model!\latest.pdparams"
    wget -q -P tmp\ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/!model!_pretrained.pdparams --no-proxy --no-check-certificate
    echo f| xcopy /s /y /F "tmp\!model!_pretrained.pdparams" "output\!model!\latest.pdparams"
    rmdir tmp /S /Q
) else (
    echo ######   not load pretrain
)

echo "RedNet50 LeViT_128S GhostNet_x1_3 RedNet50 TNT_small LeViT_128S SwinTransformer_large_patch4_window12_384"| findstr !model_latest! >nul
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

@REM # 判断是单独评估还是训练后评估
if exist %output_dir%/%model_name% (
    %output_dir%/%model_name%
    set pretrained_model=%output_dir%/%model_name%/%model_latest%/latest
) else (
    set pretrained_model=null
)

@REM # export_model
python tools/export_model.py -c  %1 -o Global.pretrained_model=%pretrained_model% -o Global.save_inference_dir=./inference/!model_name! -o Global.device=%set_cuda_device% >%log_path%/!model_name!_export_model.log 2>&1

if not !errorlevel! == 0 (
        type   "%log_path%\!model!_export.log"
        echo   !model!,export_model,FAIL  >> %log_path%\result.log
        echo  export_modeling of !model! failed!
        echo "export_exit_code: 1.0" >> %log_path%\!model!_export.log
) else (
        echo   !model!,export_model,SUCCESS  >> %log_path%\result.log
        echo   export_model of !model! successfully!
        echo "export_exit_code: 0.0" >> %log_path%\!model!_export.log
)
