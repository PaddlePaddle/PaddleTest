@REM # 输入变量：yaml、设置卡数 CPU SET_CUDA SET_MULTI_CUDA 、训练的模型动态图/静态图/收敛性( dynamic static convergence )

set sed="C:\Program Files\Git\usr\bin\sed.exe"
setlocal enabledelayedexpansion
CALL prepare.bat %1 %2

cd %Project_path%
@REM  #确定下执行路径


echo "GoogLeNet VGG11 ViT_small_patch16_224 PPLCNet_x1_0 MobileNetV3_large_x1_0 RedNet50 TNT_small LeViT_128S GhostNet_x1_3"| findstr %model_latest% >nul
if !errorlevel! equ 0 (
	%sed% -i s/"learning_rate:"/"learning_rate: 0.0001 #"/g %1
	echo "change lr"
)

set common_par="-o -o Global.epochs=1 -o DataLoader.Train.sampler.batch_size=1 -o DataLoader.Eval.sampler.batch_size=1 -o Global.seed=1234  -o DataLoader.Train.loader.num_workers=0 -o DataLoader.Train.sampler.shuffle=False -o Global.eval_interval=1 -o Global.save_interval=1 -o Global.output_dir=%output_dir%/%model_name% -o Global.device=%set_cuda_device%"
python tools/train.py -c %1 %common_par% > %log_path%\%model_name%_train.log 2>&1

if not !errorlevel! == 0 (
        type   "%log_path%\%model%_train.log"
        echo   %model%,train,FAIL  >> %log_path%\result.log
        echo  training of %model% failed!
        echo "training_exit_code: 1.0" >> %log_path%\%model%_train.log
) else (
        echo   %model%,train,SUCCESS  >> %log_path%\result.log
        echo   training of %model% successfully!
        echo "training_exit_code: 0.0" >> %log_path%\%model%_train.log
)
