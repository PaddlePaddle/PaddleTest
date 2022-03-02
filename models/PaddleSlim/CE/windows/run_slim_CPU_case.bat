@echo off

::存放 PaddleSlim repo代码
if exist ./repos rd /s /q repos
mkdir repos && cd repos
set repo_path=%cd%
echo %repo_path%
cd ..

::下载数据集
wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz --no-check-certificate
tar xf ILSVRC2012_data_demo.tar.gz

wget https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/cifar-100-python.tar.gz --no-check-certificate
tar xf cifar-100-python.tar.gz

wget -q https://sys-p0.bj.bcebos.com/slim_ci/word_2evc_demo_data.tar.gz --no-check-certificate
tar xf word_2evc_demo_data.tar.gz

set data_path=%cd%
echo %data_path%

:: log文件统一存储
if exist ./logs rd /s /q logs
mkdir logs && cd logs
set log_path=%cd%
echo %log_path%
cd ..

::下载预训练模型
set root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
set pre_models="MobileNetV1 MobileNetV2 MobileNetV3_large_x1_0_ssld ResNet101_vd MobileNetV2 ResNet34 ResNet50 ResNet50_vd"
if exist ./pretrain rd /s /q pretrain
mkdir pretrain && cd pretrain

setlocal enabledelayedexpansion
for %%i in (MobileNetV1 MobileNetV2 MobileNetV3_large_x1_0_ssld ResNet101_vd MobileNetV2 ResNet34 ResNet50 ResNet50_vd) do (
echo --------wget %%i---------------
wget -q %root_url%/%%i_pretrained.tar
tar xf %%i_pretrained.tar
)
cd ..
set pretrain_path=%cd%\pretrain
echo ----------dir------
dir

set http_proxy=%2
set https_proxy=%2


echo git clone : %1
cd %repo_path%
git clone -b %1 https://github.com/PaddlePaddle/PaddleSlim.git

echo -----------------install paddleslim-----------------
python -m pip uninstall paddleslim -y
cd %repo_path%/PaddleSlim
git branch
python -m pip install -r requirements.txt
python setup.py install
pip list

cd %repo_path%/PaddleSlim/demo
mkdir data && cd data
mklink /J ILSVRC2012 %data_path%\ILSVRC2012_data_demo\ILSVRC2012
cd ../
mklink /J pretrain %pretrain_path%
dir

rem if %2 == True (
rem     call :run_UT
rem     )

rem :run_UT
rem echo ----------- run UT -----------
rem cd %repo_path%/PaddleSlim/tests
rem call :ut_test
rem move ut_logs %log_path%\st_ut_logs

rem cd %repo_path%/PaddleSlim/tests/dygraph
rem call :ut_test
rem move ut_logs %log_path%\st_ut_logs
rem goto :eof

rem :ut_test
rem if exist ./ut_logs rd /s /q ut_logs
rem mkdir ut_logs
rem setlocal enabledelayedexpansion
rem for  %%i in (test_*.py) do ( 
rem echo %%i 
rem python %%i > ut_logs\%%~ni.txt 2>&1 
rem if %errorlevel% == 1 (
rem     move ut_logs\%%~ni.txt ut_logs\F_%%~ni.txt
rem ) else (
rem     move ut_logs\%%~ni.txt ut_logs\S_%%~ni.txt )
rem )
rem goto :eof

echo -----------------run P0case:start ------------------------

echo 1 distillion
call :all_distillation

echo 2 quant
call :all_quant

echo 3 prune
call :all_prune

echo 4 nas
call :all_nas

::echo 5 darts
::call :darts_1

rem echo 6 dygraph_qat
rem call :all_dygraph_qat

cd %log_path%
for /f "delims=" %%i in (' find /C "FAIL" result.log ') do set result=%%i
echo %result:~-1%

for /f "delims=" %%i in (' echo %result:~-1% ') do set exitcode=%%i
echo -----fail case:%exitcode%---------
echo -----exit code:%exitcode%---------
exit %exitcode%

echo -----------------run P0case:end -------------------------
goto :eof


::distillion
:all_distillation
call :dist_res50_v1
::call :dist_MobileNetV2_x0_25
call :dml_mv1_mv1
::call :dml_mv1_res50
goto :eof

:dist_res50_v1
cd %repo_path%/PaddleSlim/demo/distillation
set model=dist_res50_v1
python distill.py  --batch_size 64 --use_gpu False ^
--total_images 1000 --image_shape 3,224,224 --lr 0.1 ^
--lr_strategy piecewise_decay --l2_decay 3e-05 --momentum_rate 0.9 ^
--num_epochs 1 --data imagenet --log_period 20 --model MobileNet ^
--teacher_model ResNet50_vd --teacher_pretrained_model ../pretrain/ResNet50_vd_pretrained ^
--step_epochs 30 60 90 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:: add MobileNetV2_x0_25
:dist_MobileNetV2_x0_25
cd %repo_path%/PaddleSlim/demo/distillation
set model=dist_MobileNetV2_x0_25
python distill.py --num_epochs 1 --batch_size 64 --save_inference True ^
--model MobileNetV2_x0_25 ^
--teacher_model MobileNetV2 --teacher_pretrained_model ../pretrain/MobileNetV2_pretrained > %log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:dml_mv1_mv1
cd %repo_path%/PaddleSlim/demo/deep_mutual_learning
mkdir dataset && cd dataset
mklink /J cifar %data_path%\cifar-100-python
cd ..
set model=dml_mv1_mv1
python dml_train.py  --epochs 1 --batch_size 32 --use_gpu=False >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:dml_mv1_res50
cd %repo_path%/PaddleSlim/demo/deep_mutual_learning
set model=dml_mv1_res50
python dml_train.py --models="mobilenet-resnet50" --epochs 1 --batch_size 32 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

::quant
:all_quant
call :quant_aware
call :st_quant_post
call :pact_quant_aware
call :dy_quant
call :quant_embedding
goto :eof

:quant_aware
call :quant_aware_v1_MobileNet
::call :quant_aware_v2_ResNet34
goto :eof

:quant_aware_v1_MobileNet
cd %repo_path%/PaddleSlim/demo/quant/quant_aware
set model=quant_aware_v1_MobileNet
python train.py --model ResNet34 ^
--pretrained_model ../../pretrain/ResNet34_pretrained --use_gpu=False ^
--checkpoint_dir ./output/ResNet34 --num_epochs 1 --batch_size 32 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

::add ResNet34
:quant_aware_v2_ResNet34
cd %repo_path%/PaddleSlim/demo/quant/quant_aware
set model=quant_aware_v2_ResNet34
python train.py --model MobileNet --pretrained_model ../../pretrain/MobileNetV1_pretrained ^
--checkpoint_dir ./output/mobilenetv1 --num_epochs 1 --batch_size 32 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

::add quant_embedding
:quant_embedding
cd %repo_path%/PaddleSlim/demo/quant/quant_embedding
mklink /J data %data_path%\word_2evc_demo_data
set OPENBLAS_NUM_THREADS=1 
set CPU_NUM=5
set model=quant_em_word2vec
python train.py --train_data_dir data/convert_text8 ^
    --dict_path data/test_build_dict --num_passes 1 --batch_size 100 ^
    --model_output_dir v1_cpu5_b100_lr1dir ^
    --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof  

:st_quant_post
call :quant_post_v1_export
call :quant_post_hist
goto :eof 

:quant_post_v1_export
cd %repo_path%/PaddleSlim/demo/quant/quant_post
:: 导出模型 
echo run quant_post_v1_export
set model=quant_post_v1_export
python export_model.py --model "MobileNet" --use_gpu=False ^
--pretrained_model ../../pretrain/MobileNetV1_pretrained ^
--data imagenet >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:quant_post_hist
cd %repo_path%/PaddleSlim/demo/quant/quant_post
setlocal enabledelayedexpansion
for %%A in (hist) do (
echo run %%A 
::不带bc的离线量化
set model=st_quant_post_v1_T_%%A
echo run !model!
python quant_post.py --model_path ./inference_model/MobileNet ^
--save_path ./quant_model/%%A/MobileNet --use_gpu False ^
--model_filename model --params_filename weights --algo %%A >%log_path%\!model!.log 2>&1
call :printInfo  %errorlevel%

::量化后eval
set model=st_quant_post_%%A_eval2
echo run !model!
python eval.py --model_path ./quant_model/%%A/MobileNet --model_name __model__ --use_gpu False ^
--params_name __params__ >%log_path%\!model!.log 2>&1
call :printInfo  %errorlevel%

::带bc参数的的离线量化
set model=st_quant_post_T_%%A_bc
echo run !model!
set bc_path=%%A_bc
python quant_post.py --model_path ./inference_model/MobileNet ^
--save_path ./quant_model/!bc_path!/MobileNet --use_gpu False ^
--model_filename model --params_filename weights ^
--algo %%A --bias_correction True >%log_path%\!model!.log 2>&1
call :printInfo  %errorlevel%

::量化后eval
set model=st_quant_post_%%A_bc_eval2
echo run !model!
python eval.py --model_path ./quant_model/!bc_path!/MobileNet --model_name __model__ ^
--params_name __params__ --use_gpu False >%log_path%\!model!.log 2>&1
call :printInfo  %errorlevel%
)
goto :eof

::add pact_quant_aware
:pact_quant_aware
::call :pact_quant_aware_not_usepact
call :pact_quant_aware_usepact
call :pact_quant_aware_load
goto :eof

:pact_quant_aware_not_usepact
cd %repo_path%/PaddleSlim/demo/quant/pact_quant_aware
set model=pact_quant_aware_not_usepact
cd %repo_path%/PaddleSlim/demo/quant/pact_quant_aware
python train.py --model MobileNetV3_large_x1_0 ^
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained ^
--num_epochs 1 --lr 0.0001 --use_pact False --batch_size 32 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:pact_quant_aware_usepact
cd %repo_path%/PaddleSlim/demo/quant/pact_quant_aware
set model=pact_quant_aware_usepact
python train.py --model MobileNetV3_large_x1_0 ^
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained --use_gpu False ^
--num_epochs 1 --lr 0.0001 --use_pact True --batch_size 32 --lr_strategy=piecewise_decay ^
--step_epochs 2 --l2_decay 1e-5 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

::load
:pact_quant_aware_load
cd %repo_path%/PaddleSlim/demo/quant/pact_quant_aware
set model=pact_quant_aware_load
python train.py --model MobileNetV3_large_x1_0 ^
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained ^
--num_epochs 2 --lr 0.0001 --use_pact True --batch_size 32 --lr_strategy=piecewise_decay ^
--step_epochs 20 --l2_decay 1e-5 ^
--checkpoint_dir ./output/MobileNetV3_large_x1_0/0 ^
--checkpoint_epoch 0 --use_gpu False >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:: add ce_tests

:all_dygraph_qat
call :ce_tests_dygraph_qat1
::call :ce_tests_dygraph_qat4
goto :eof

:ce_tests_dygraph_qat1
cd %repo_path%/PaddleSlim/ce_tests/dygraph/qat
set test_samples=1000
set data_path_in="./ILSVRC2012/"
set batch_size=16
set epoch=1
set lr=0.0001
set num_workers=1
set output_dir="./output_models"
setlocal enabledelayedexpansion
mklink /J ILSVRC2012 %data_path%\ILSVRC2012_data_demo\ILSVRC2012
for %%M in (mobilenet_v1) do (
    ::1 quant train
    rem echo "------1 nopact train--------", %%M
    rem set model=qat_%%M_gpu1_nw1_1
    rem python ./src/qat.py ^
    rem     --arch=%%M ^
    rem     --data=%data_path_in% ^
    rem     --epoch=%epoch% ^
    rem     --batch_size=32 ^
    rem     --num_workers=%num_workers% ^
    rem     --lr=%lr%^
    rem     --output_dir=%output_dir% ^
    rem     --enable_quant >%log_path%\!model!.log 2>&1
    rem call :printInfo  %errorlevel%
    rem echo "------ 2 eval before save quant ----------", %%M
    rem set model=eval_before_save_%%M_1
    rem python ./src/eval.py ^
    rem     --model_path=./output_models/quant_dygraph/%%M ^
    rem     --data_dir=%data_path_in% ^
    rem     --test_samples=%test_samples% ^
    rem     --batch_size=%batch_size% ^ >%log_path%\!model!.log 2>&1
    rem call :printInfo  %errorlevel%
    rem echo "--------3 save_nopact_quant_model----------", %%M
    rem set model=save_quant_%%M_1
    rem python src/save_quant_model.py ^
    rem       --load_model_path ./output_models/quant_dygraph/%%M ^
    rem       --save_model_path ./int8_models/%%M > %log_path%\!model!.log 2>&1
    rem call :printInfo  %errorlevel%
    rem echo "--------4 CPU eval after save nopact quant ------", %%M
    rem set model=cpu_eval_after_save_%%M_1
    rem python ./src/eval.py ^
    rem     --model_path=./int8_models/%%M ^
    rem     --data_dir=%data_path_in% ^
    rem     --test_samples=%test_samples% ^
    rem     --batch_size=%batch_size% >%log_path%\!model!.log 2>&1
    rem call :printInfo  %errorlevel%

    ::pact quant train
    echo "------1 pact train--------", %%M
    set model=pact_qat_%%M_gpu1_nw1
    python ./src/qat.py ^
        --arch=%%M ^
        --data=%data_path_in% ^
        --epoch=%epoch% ^
        --batch_size=32 ^
        --num_workers=%num_workers% ^
        --lr=%lr% ^
        --output_dir=./output_models_pact/ ^
        --enable_quant ^
        --use_pact >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
    echo "--------2 eval before save pact quant --------", %%M
    set model=val_before_pact_save_%%M
    python ./src/eval.py ^
        --model_path=./output_models_pact/quant_dygraph/%%M ^
        --data_dir=%data_path_in% ^
        --test_samples=%test_samples% ^
        --batch_size=%batch_size% >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
    echo "--------3  save pact quant -------------", %%M
    set model=save_pact_quant_%%M
    python src/save_quant_model.py ^
          --load_model_path ./output_models_pact/quant_dygraph/%%M ^
          --save_model_path ./int8_models_pact/%%M >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
    echo "--------4 CPU eval after save pact quant --------", %%M
    set model=cpu_eval_after_pact_save_%%M
    python ./src/eval.py ^
        --model_path=./int8_models_pact/%%M ^
        --data_dir=%data_path_in% ^
        --test_samples=%test_samples% ^
        --batch_size=%batch_size% >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
)
goto :eof

:ce_tests_dygraph_qat4
cd %repo_path%/PaddleSlim/ce_tests/dygraph/qat
set test_samples=1000
set data_path_in="./ILSVRC2012/"
set batch_size=16
set epoch=1
set lr=0.0001
set num_workers=1
set output_dir="./output_models_4"
mklink /J ILSVRC2012 %data_path%\ILSVRC2012_data_demo\ILSVRC2012
setlocal enabledelayedexpansion
for %%M in (vgg16) do (
    ::1 quant train
    echo "------1 nopact train--------", %%M
    set model=qat_%%M_gpu1_nw1
    python ./src/qat.py ^
        --arch=%%M ^
        --data=%data_path_in% ^
        --epoch=%epoch% ^
        --batch_size=32 ^
        --num_workers=%num_workers% ^
        --lr=%lr%^
        --output_dir=%output_dir% ^
        --enable_quant >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
    echo "------ 2 eval before save quant ----------", %%M
    set model=eval_before_save_%%M
    python ./src/eval.py ^
        --model_path=%output_dir%/quant_dygraph/%%M ^
        --data_dir=%data_path_in% ^
        --test_samples=%test_samples% ^
        --batch_size=%batch_size% ^ >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
    echo "--------3 save_nopact_quant_model----------", %%M
    set model=save_quant_%%M
    python src/save_quant_model.py ^
          --load_model_path %output_dir%/quant_dygraph/%%M ^
          --save_model_path ./int8_models/%%M > %log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
    echo "--------4 CPU eval after save nopact quant ------", %%M
    set model=cpu_eval_after_save_%%M
    set CUDA_VISIBLE_DEVICES=
    python ./src/eval.py ^
        --model_path=./int8_models/%%M ^
        --data_dir=%data_path_in% ^
        --test_samples=%test_samples% ^
        --batch_size=%batch_size% >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%

    ::pact quant train
    echo "------1 pact train--------", %%M
    set model=pact_qat_%%M_gpu1_nw1
    python ./src/qat.py ^
        --arch=%%M ^
        --data=%data_path_in% ^
        --epoch=%epoch% ^
        --batch_size=16 ^
        --num_workers=%num_workers% ^
        --lr=%lr% ^
        --output_dir=./output_models_4_pact/ ^
        --enable_quant ^
        --use_pact >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
    echo "--------2 eval before save pact quant --------", %%M
    set model=val_before_pact_save_%%M
    python ./src/eval.py ^
        --model_path=./output_models_4_pact/quant_dygraph/%%M ^
        --data_dir=%data_path_in% ^
        --test_samples=%test_samples% ^
        --batch_size=%batch_size% >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
    echo "--------3  save pact quant -------------", %%M
    set model=save_pact_quant_%%M
    python src/save_quant_model.py ^
          --load_model_path ./output_models_4_pact/quant_dygraph/%%M ^
          --save_model_path ./int8_models_pact/%%M >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
    echo "--------4 CPU eval after save pact quant --------", %%M
    set CUDA_VISIBLE_DEVICES=
    set model=cpu_eval_after_pact_save_%%M
    python ./src/eval.py ^
        --model_path=./int8_models_pact/%%M ^
        --data_dir=%data_path_in% ^
        --test_samples=%test_samples% ^
        --batch_size=%batch_size% >%log_path%\!model!.log 2>&1
    call :printInfo  %errorlevel%
)
goto :eof

:dy_quant
call :dy_quant_v1
::call :dy_pact_quant_v3
goto :eof

:dy_quant_v1
cd %repo_path%/PaddleSlim/demo/dygraph/quant
set model=dy_quant_v1
python train.py --model=mobilenet_v1 --use_gpu False ^
--pretrained_model '../../pretrain/MobileNetV1_pretrained' ^
--num_epochs 1 --batch_size 32 > %log_path%/%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:dy_pact_quant_v3
set model=dy_pact_quant_v3
python train.py  --lr=0.001 --batch_size 32 --use_pact=True --num_epochs=2 --l2_decay=2e-5 ^
    --ls_epsilon=0.1  --pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained  ^
    > %log_path%/%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

::prune

:all_prune
call :prune_v1
call :prune_fpgm_MobileNetV1
::call :prune_fpgm_MobileNetV2
::call :prune_fpgm_resnet34_42
::call :prune_fpgm_resnet34_50
::call :prune_ResNet50
call :dy_prune_ResNet34_f42

call :unstructured_prune
goto :eof

:unstructured_prune
call :st_unstructured_prune
call :dy_unstructured_prune
goto :eof

:prune_v1
cd %repo_path%/PaddleSlim/demo/prune
set model=prune_v1
python train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" --use_gpu False ^
--num_epochs 2 --batch_size 64 --pretrained_model="../pretrain/MobileNetV1_pretrained" ^
--save_inference True --test_period=1 > %log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

rem set model=prune_v1_eval
rem python eval.py --model "MobileNet" --data "imagenet" --model_path "./models/0" >%log_path%\%model%.log 2>&1
rem call :printInfo  %errorlevel%

::add prune_fpgm MobileNet
:prune_fpgm_MobileNetV1
cd %repo_path%/PaddleSlim/demo/prune
set model=prune_fpgm_MobileNetV1
python train.py --model="MobileNet" --pretrained_model="../pretrain/MobileNetV1_pretrained" ^
    --data="imagenet" --pruned_ratio=0.3125 --lr=0.1 --num_epochs=1 --test_period=1 ^
    --step_epochs 30 60 90 --batch_size 64 --use_gpu False ^
    --l2_decay=3e-5 --lr_strategy="piecewise_decay" ^ --criterion="geometry_median" ^
    --model_path="./fpgm_mobilenetv1_models" --save_inference True  >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

::add prune_fpgm MobileNetV2
:prune_fpgm_MobileNetV2
cd %repo_path%/PaddleSlim/demo/prune
set model=prune_fpgm_MobileNetV2
python train.py --model="MobileNetV2" --pretrained_model="../pretrain/MobileNetV2_pretrained" ^
    --data="imagenet" --pruned_ratio=0.325 --lr=0.001 --num_epochs=2 --test_period=1 ^
    --step_epochs 30 60 80 --l2_decay=1e-4 --lr_strategy="piecewise_decay" ^
    --criterion="geometry_median" --model_path="./output/fpgm_mobilenetv2_models" ^
    --save_inference True --batch_size 64 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:prune_fpgm_resnet34_42
cd %repo_path%/PaddleSlim/demo/prune
set model=prune_fpgm_resnet34_42
python train.py --model="ResNet34" --pretrained_model="../pretrain/ResNet34_pretrained" ^
    --data="imagenet" --pruned_ratio=0.25  --num_epochs=2 --test_period=1 ^
    --lr_strategy="piecewise_decay" ^
    --criterion="geometry_median" --model_path="./output/fpgm_resnet34_025_120_models" ^
    --save_inference True >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
set model=slim_prune_fpgm_resnet34_42_eval
python eval.py --model "ResNet34" --data "imagenet" ^
--model_path "./output/fpgm_resnet34_025_120_models/0" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:prune_fpgm_resnet34_50
set model=prune_fpgm_resnet34_50
cd %repo_path%/PaddleSlim/demo/prune
python train.py --model="ResNet34" --pretrained_model="../pretrain/ResNet34_pretrained" ^
    --data="imagenet" --pruned_ratio=0.3125 --lr=0.001 --num_epochs=2 --test_period=1 ^
    --step_epochs 30 60 --l2_decay=1e-4 --lr_strategy="piecewise_decay" --criterion="geometry_median" ^
    --model_path="./output/fpgm_resnet34_50_models" ^
    --save_inference True >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
set model=slim_prune_fpgm_resnet34_50_eval
python eval.py --model "ResNet34" --data "imagenet" ^
    --model_path "./output/fpgm_resnet34_50_models/0" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:prune_ResNet50
set model=prune_ResNet50
cd %repo_path%/PaddleSlim/demo/prune
python train.py --model ResNet50 --pruned_ratio 0.31 --data "imagenet" ^
    --save_inference True --pretrained_model ../pretrain/ResNet50_pretrained ^
    --num_epochs 1 --batch_size 64 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:dy_prune_ResNet34_f42
::train--恢复train--eval--export
set model=dy_prune_ResNet34_f42
cd %repo_path%/PaddleSlim/demo/dygraph/pruning
mkdir data && cd data
mklink /J ILSVRC2012 %data_path%\ILSVRC2012_data_demo\ILSVRC2012
cd ..
python train.py --use_gpu=False --model="resnet34" --data="imagenet" ^
    --pruned_ratio=0.25 --num_epochs=1 --batch_size=128 --lr_strategy="cosine_decay" ^
    --criterion="fpgm" ^
    --model_path="./fpgm_resnet34_025_120_models" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
set model=dy_prune_ResNet50_f42_gpu1_load
python train.py --use_gpu=False --model="resnet34" --data="imagenet" --pruned_ratio=0.25 ^
    --num_epochs=2 --batch_size=128 --lr_strategy="cosine_decay" --criterion="fpgm" ^
    --model_path="./fpgm_resnet34_025_120_models" ^
    --checkpoint="./fpgm_resnet34_025_120_models/0" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

set model=dy_prune_ResNet50_f42_gpu1_eval
python eval.py --checkpoint=./fpgm_resnet34_025_120_models/1 --model="resnet34" ^
    --pruned_ratio=0.25 --batch_size=128 --use_gpu=False >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

set model=dy_prune_ResNet50_f42_gpu1_export
python export_model.py --checkpoint=./fpgm_resnet34_025_120_models/final ^
    --model="resnet34" --pruned_ratio=0.25 --output_path=./infer_final/resnet >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:st_unstructured_prune
set model=st_unstructured_prune_threshold
echo run st_unstructured_prune_threshold
cd %repo_path%/PaddleSlim/demo/unstructured_prune
python train.py --batch_size 64 --pretrained_model ../pretrain/MobileNetV1_pretrained ^
    --lr 0.05 --pruning_mode threshold --threshold 0.01 --data imagenet --use_gpu=False ^
    --lr_strategy piecewise_decay --step_epochs 1 2 3 --num_epochs 5 --model_period 2 ^
    --test_period 1 ^
     --model_path ./st_unstructured_models >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

echo run st_unstructured_prune_threshold_eval
set model=st_unstructured_prune_threshold_eval
python evaluate.py --pruned_model=./st_unstructured_models ^
       --data="imagenet" --use_gpu=False >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

echo run st_unstructured_prune_threshold_load
set model=st_unstructured_prune_threshold_load
::python train.py ^
::    --batch_size 128 --pretrained_model ../pretrain/MobileNetV1_pretrained --use_gpu=False ^
::    --lr 0.05 --pruning_mode threshold ^--threshold 0.01 --data imagenet ^
::   --lr_strategy piecewise_decay --step_epochs 1 2 3 --num_epochs 3 --test_period 1 ^
::    --model_path ./st_unstructured_models ^
::    --last_epoch 1 >%log_path%\%model%.log 2>&1
::call :printInfo  %errorlevel%

echo run st_ratio_prune_ratio
set model=st_unstructured_prune_ratio
python train.py --batch_size 64 --pretrained_model ../pretrain/MobileNetV1_pretrained ^
    --lr 0.05 --pruning_mode ratio --ratio 0.55 --data imagenet --use_gpu=False ^
    --lr_strategy piecewise_decay --step_epochs 1 2 3 --num_epochs 1 --test_period 1 ^
    --model_path st_ratio_models >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

echo run st_unstructured_prune_threshold_mnist_T
set model=st_unstructured_prune_threshold_mnist_T
python train.py --batch_size 256 --pretrained_model ../pretrain/MobileNetV1_pretrained ^
    --lr 0.05 --pruning_mode threshold --threshold 0.01 --data mnist --use_gpu=False ^
    --lr_strategy piecewise_decay --step_epochs 1 2 3 --num_epochs 1 --test_period 1 ^
    --model_path st_unstructured_models_mnist >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
rem set model=st_unstructured_models_mnist
rem python evaluate.py --pruned_model=st_unstructured_models_mnist ^
rem      --data="mnist"  >%log_path%\%model%.log 2>&1
rem call :printInfo  %errorlevel%
goto :eof

:dy_unstructured_prune
echo run dy_ratio_prune_ratio_T
set model=dy_ratio_prune_ratio_T
cd %repo_path%/PaddleSlim/demo/dygraph/unstructured_pruning
python train.py ^
--data imagenet ^
--lr 0.05 ^
--pruning_mode ratio ^
--ratio 0.55 ^
--batch_size 64 ^
--lr_strategy piecewise_decay ^
--step_epochs 1 2 3 ^
--num_epochs 5 --model_period 2 ^
--test_period 1 --use_gpu=False ^
--model_path "./dy_threshold_models" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

set model=dy_threshold_prune_T
python  train.py ^
--data imagenet ^
--lr 0.05 ^
--pruning_mode threshold ^
--threshold 0.01 ^
--batch_size 64 ^
--lr_strategy piecewise_decay ^
--step_epochs 1 2 3 ^
--num_epochs 5 --model_period 2 ^
--test_period 1 --use_gpu=False ^
--model_path "./dy_threshold_models" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

set model=dy_threshold_prune_eval
python evaluate.py --pruned_model "./dy_threshold_models/model.pdparams" ^
--data imagenet >%log_path%\%model%.log --use_gpu=False 2>&1
call :printInfo  %errorlevel%

rem set model=dy_threshold_prune_T_load
rem python -m paddle.distributed.launch ^
rem --log_dir train_dy_ratio_log train.py ^
rem --data imagenet ^
rem --lr 0.05 ^
rem --pruning_mode threshold ^
rem --threshold 0.01 ^
rem --batch_size 64 ^
rem --lr_strategy piecewise_decay ^
rem --step_epochs 1 2 3 ^
rem --num_epochs 3 ^
rem --test_period 1 ^
rem --model_path "./dy_threshold_models" ^
rem --pretrained_model "./dy_threshold_models/model.pdparams" ^
rem --last_epoch 1 >%log_path%\%model%.log 2>&1
rem call :printInfo  %errorlevel%

set model=dy_threshold_prune_cifar10_T
python train.py --data cifar10 --lr 0.05 ^
--pruning_mode threshold --num_epochs 1 --use_gpu=False ^
--threshold 0.01 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:all_nas
call :sa_nas_v2_T_1card
rem call :block_sa_nas_v2_T_1card
rem call :rl_nas_v2_T_1card
goto :eof

::nas
:sa_nas_v2_T_1card
cd %repo_path%/PaddleSlim/demo/nas
set model=sa_nas_v2_T_1card
python sa_nas_mobilenetv2.py --search_steps 1 --use_gpu False >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:block_sa_nas_v2_T_1card
cd %repo_path%/PaddleSlim/demo/nas
set model=block_sa_nas_v2_T_1card
python block_sa_nas_mobilenetv2.py --search_steps 1 --use_gpu=False >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:rl_nas_v2_T_1card
cd %repo_path%/PaddleSlim/demo/nas
set model=rl_nas_v2_T_1card
python rl_nas_mobilenetv2.py --search_steps 1 --port 8885 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:all_darts
call :darts_1
goto :eof

:darts_1
cd %repo_path%/PaddleSlim/demo/darts
set model=darts1_search_1card
python search.py --epochs 1 --use_multiprocess False --batch_size 16 --use_gpu=False  >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

rem set model=pcdarts_train_1card
rem python train.py --arch "PC_DARTS" --epochs 1 --use_multiprocess False --batch_size 32 >%log_path%\%model%.log 2>&1
rem call :printInfo  %errorlevel%
goto :eof

:printInfo
if %1 == 1 (
    move %log_path%\%model%.log %log_path%\FAIL_%model%.log
    echo  FAIL_%model%.log >> %log_path%\result.log
) else (
    move %log_path%\%model%.log %log_path%\SUCCESS_%model%.log
    echo SUCCESS_%model%.log >> %log_path%\result.log
)
goto :eof
