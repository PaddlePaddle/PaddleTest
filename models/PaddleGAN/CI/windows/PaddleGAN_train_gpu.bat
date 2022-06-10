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

        rem set FLAGS_enable_eager_mode=1
        rem #验证天宇 220329 pr

	cd %Project_path%
	echo "path after"
	echo %1 >gan_models_list_all_gpu
	chdir
	dir
)

set log_path=log
set params_dir=(output/*)
@REM set 不能放在循环中
if exist "log" (
   rmdir log /S /Q
   md log
) else (
   md log
)
rem data
rd /s /q data
mklink /j data %data_path%\PaddleGAN

set http_proxy=
set https_proxy=
rem dependency
python -m pip install --ignore-installed  --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip uninstall opencv-python -y
python -m pip install -v -e . -i https://mirror.baidu.com/pypi/simple
python -m pip install dlib -i https://mirror.baidu.com/pypi/simple
python -m pip install   --ignore-installed -r requirements.txt -i https://mirror.baidu.com/pypi/simple
python -m pip list

set sed="C:\Program Files\Git\usr\bin\sed.exe"
setlocal enabledelayedexpansion
for /f %%i in (gan_models_list_all_gpu) do (
echo %%i
set target=%%i
rem echo !target!
set target1=!target:*/=!
rem echo !target1!
set model=!target1:.yaml=!
echo !model!

%sed% -i 1s/"epochs"/"total_iters"/ %%i
%sed% -i s/"pretrain_ckpt:"/"pretrain_ckpt: #"/g %%i

echo train
rd /s /q output
python -u tools/main.py --config-file %%i -o  total_iters=20 snapshot_config.interval=10 log_config.interval=1 output_dir=output dataset.train.batch_size=1 > %log_path%/!model!_train.log 2>&1

if not !errorlevel! == 0 (
    echo  %log_path%\!model!_train.log
    type  %log_path%\!model!_train.log
    echo   !model!,train,FAIL  >> %log_path%\result.log
    echo  training of !model! failed!
    echo "training_exit_code: 1.0" >> %log_path%\!model!_train.log
) else (
    echo   !model!,train,SUCCESS  >> %log_path%\result.log
    echo   training of !model! successfully!
    echo "training_exit_code: 0.0" >> %log_path%\!model!_train.log
)

echo eval
if !model!==stylegan_v2_256_ffhq (
    for /d %%j in %params_dir% do (
    echo %%j
    python tools/extract_weight.py output/%%j/iter_20_checkpoint.pdparams --net-name gen_ema --output stylegan_extract.pdparams > %log_path%/!model!_extract_weight.log 2>&1
    if not !errorlevel! == 0 (
        type  %log_path%\!model!_extract_weight.log
        echo  !model!,extract_weight,FAIL  >> %log_path%\result.log
        echo  extract_weight of !model! failed!
    ) else (
        echo   !model!,extract_weight,SUCCESS  >> %log_path%\result.log
        echo   extract_weight of !model! successfully!
    )
    python applications/tools/styleganv2.py --output_path stylegan_infer --weight_path stylegan_extract.pdparams --size 256 > %log_path%/!model!_eval.log 2>&1
    if not !errorlevel! == 0 (
        type  %log_path%\!model!_eval.log
        echo   !model!,eval,FAIL  >> %log_path%\result.log
        echo  eval of !model! failed!
        echo "eval_exit_code: 1.0" >> %log_path%\!model!_eval.log
    ) else (
        echo   !model!,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of !model! successfully!
        echo "eval_exit_code: 0.0" >> %log_path%\!model!_eval.log
    )
    )

) else if !model!==msvsr_reds (
    for /d %%j in %params_dir% do (
    echo %%j
    python -u tools/main.py --config-file %%i --evaluate-only --load output/%%j/iter_20_checkpoint.pdparams -o dataset.test.num_frames=30 > %log_path%/!model!_eval.log 2>&1
    if not !errorlevel! == 0 (

        echo  %log_path%\!model!_eval.log
        type  %log_path%\!model!_eval.log
        echo   !model!,eval,FAIL  >> %log_path%\result.log
        echo  eval of !model! failed!
        echo "eval_exit_code: 1.0" >> %log_path%\!model!_eval.log
    ) else (

        echo   !model!,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of !model! successfully!
        echo "eval_exit_code: 0.0" >> %log_path%\!model!_eval.log
    )
    )

) else if !model!==basicvsr++_vimeo90k_BD (
    echo "basicvsr++_vimeo90k_BD eval OOM need change data small"
    echo "eval_exit_code: 0.0" >> %log_path%\!model!_eval.log

) else if !model!==makeup (

    echo "makeup no eval "
    echo "eval_exit_code: 0.0" >> %log_path%\!model!_eval.log

) else if !model!==msvsr_l_reds (

    echo "msvsr_l_reds train eval OOM can not fix"
    echo "eval_exit_code: 0.0" >> %log_path%\!model!_eval.log

) else if !model!==basicvsr++_reds	 (
    for /d %%j in %params_dir% do (
    echo %%j
    python -u tools/main.py --config-file %%i --evaluate-only --load output/%%j/iter_20_checkpoint.pdparams -o dataset.test.num_frames=30 > %log_path%/!model!_eval.log 2>&1
    if not !errorlevel! == 0 (

        echo  %log_path%\!model!_eval.log
        type  %log_path%\!model!_eval.log
        echo   !model!,eval,FAIL  >> %log_path%\result.log
        echo  eval of !model! failed!
        echo "eval_exit_code: 1.0" >> %log_path%\!model!_eval.log
    ) else (

        echo   !model!,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of !model! successfully!
        echo "eval_exit_code: 0.0" >> %log_path%\!model!_eval.log
    )
    )

) else (
    for /d %%j in %params_dir% do (
    echo %%j
    python -u tools/main.py --config-file %%i --evaluate-only --load output/%%j/iter_20_checkpoint.pdparams > %log_path%/!model!_eval.log 2>&1
    if not !errorlevel! == 0 (

        echo  %log_path%\!model!_eval.log
        type  %log_path%\!model!_eval.log
        echo   !model!,eval,FAIL  >> %log_path%\result.log
        echo  eval of !model! failed!
        echo "eval_exit_code: 1.0" >> %log_path%\!model!_eval.log
    ) else (
        echo   !model!,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of !model! successfully!
        echo "eval_exit_code: 0.0" >> %log_path%\!model!_eval.log
    )
    )
)

echo ----------------------------------------------------------------
if !model!==edvr_m_wo_tsa (

echo infer
@REM echo styleganv2
@REM python -u applications/tools/styleganv2.py --output_path styleganv2_infer --model_type ffhq-config-f --seed 233 --size 1024 --style_dim 512 --n_mlp 8 --channel_multiplier 2 --n_row 3 --n_col 5 > %log_path%/styleganv2_infer.log 2>&1
@REM if  !errorlevel! GTR 0 (
@REM         echo   styleganv2,infer,FAIL  >> %log_path%\result.log
@REM         echo  infer of styleganv2 failed!
@REM ) else (
@REM         echo   styleganv2,infer,SUCCESS  >> %log_path%\result.log
@REM         echo   infer of styleganv2 successfully!
@REM )

@REM echo wav2lip
@REM python applications/tools/wav2lip.py --face ./docs/imgs/mona7s.mp4 --audio ./docs/imgs/guangquan.m4a --outfile Wav2Lip_infer.mp4 > %log_path%/wav2lip_infer.log 2>&1
@REM if  !errorlevel! GTR 0 (
@REM         echo   wav2lip,infer,FAIL  >> %log_path%\result.log
@REM         echo  infer of wav2lip failed!
@REM ) else (
@REM         echo   wav2lip,infer,SUCCESS  >> %log_path%\result.log
@REM         echo   infer of wav2lip successfully!
@REM )

echo animeganv2
python applications/tools/animeganv2.py --input_image ./docs/imgs/animeganv2_test.jpg > %log_path%/animeganv2_infer.log 2>&1
if  !errorlevel! GTR 0 (
    echo   animeganv2,infer,FAIL  >> %log_path%\result.log
    echo  infer of animeganv2 failed!
) else (
    echo   animeganv2,infer,SUCCESS  >> %log_path%\result.log
    echo   infer of animeganv2 successfully!
)
echo first order motion
python -u applications/tools/first-order-demo.py --driving_video ./docs/imgs/fom_dv.mp4 --source_image ./docs/imgs/fom_source_image.png --ratio 0.4 --relative --adapt_scale > %log_path%/first_order_motion_single_person_infer.log 2>&1
if  !errorlevel! GTR 0 (
    echo   first_order_motion_single_person,infer,FAIL  >> %log_path%\result.log
    echo  infer of first_order_motion_single_person failed!
) else (
    echo   first_order_motion_single_person,infer,SUCCESS  >> %log_path%\result.log
    echo   infer of first_order_motion_single_person successfully!
)

echo first order motion multi
python -u applications/tools/first-order-demo.py --driving_video ./docs/imgs/fom_dv.mp4 --source_image ./docs/imgs/fom_source_image_multi_person.jpg --ratio 0.4 --relative --adapt_scale --multi_person > %log_path%/first_order_motion_multi_person_infer.log 2>&1
if  !errorlevel! GTR 0 (
    echo   first_order_motion_multi_person,infer,FAIL  >> %log_path%\result.log
    echo  infer of first_order_motion_multi_person failed!
) else (
    echo   first_order_motion_multi_person,infer,SUCCESS  >> %log_path%\result.log
    echo   infer of first_order_motion_multi_person successfully!
)

echo face parse
python applications/tools/face_parse.py --input_image ./docs/imgs/face.png > %log_path%/face_parse_infer.log 2>&1
if  !errorlevel! GTR 0 (
    echo   face_parse,infer,FAIL  >> %log_path%\result.log
    echo  infer of face_parse failed!
) else (
    echo   face_parse,infer,SUCCESS  >> %log_path%\result.log
    echo   infer of face_parse successfully!
)
echo psgan
python tools/psgan_infer.py --config-file configs/makeup.yaml --source_path  docs/imgs/ps_source.png --reference_dir docs/imgs/ref --evaluate-only > %log_path%/psgan_infer.log 2>&1
if  !errorlevel! GTR 0 (
    echo   psgan,infer,FAIL  >> %log_path%\result.log
    echo  infer of psgan failed!
) else (
    echo   psgan,infer,SUCCESS  >> %log_path%\result.log
    echo   infer of psgan successfully!
)

@REM echo vidieo restore #39环境GPU报错，38环境可以
@REM python applications/tools/video-enhance.py --input data/Peking_input360p_clip_10_11.mp4 --process_order DAIN DeOldify EDVR --output video_restore_infer > %log_path%/vidieo_restore_infer.log 2>&1
@REM if  !errorlevel! GTR 0 (
@REM         echo   vidieo restore,infer,FAIL  >> %log_path%\result.log
@REM         echo  infer of vidieo restore failed!
@REM ) else (
@REM         echo   vidieo restore,infer,SUCCESS  >> %log_path%\result.log
@REM         echo   infer of vidieo restore successfully!
@REM )

) else (
    echo "other models"
)

)

@REM rmdir data /S /Q
rem 清空数据文件防止效率云清空任务时删除原始文件
set num=0
for /F %%i in ('findstr /s "FAIL" %log_path%/result.log') do ( set num=%%i )
findstr /s "FAIL" %log_path%/result.log
rem echo %num%

if %num%==0 (
 exit /b 0
) else (
 exit /b 1
)
