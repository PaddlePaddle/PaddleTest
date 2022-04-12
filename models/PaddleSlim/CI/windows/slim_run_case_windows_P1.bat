cd %repo_path%/PaddleSlim/demo
mkdir data && cd data
mklink /J ILSVRC2012 %data_path%\ILSVRC2012_data_demo\ILSVRC2012
cd ../
mklink /J pretrain %pretrain_path%

echo -----------------run P1case:start ------------------------

echo 1 distillion
call :all_distillation

echo 2 quant
call :all_quant

echo 3 prune
call :all_prune

echo 4 nas
call :all_nas

echo -----------------run P1case:end -------------------------

::distillion
:all_distillation
call :dist_MobileNetV2_x0_25
::call :dml_mv1_mv1
::call :dml_mv1_res50
goto :eof


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
python dml_train.py  --epochs 1 --batch_size 32 --use_gpu True >%log_path%\%model%.log 2>&1
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
call :st_quant_aware_v2_ResNet34
call :st_pact_quant_aware
call :dy_quant
call :dy_quant_v1
goto :eof

:st_quant_aware_v2_ResNet34
cd %repo_path%/PaddleSlim/demo/quant/quant_aware
set model=st_quant_aware_v2_ResNet34
python train.py --model MobileNet --pretrained_model ../../pretrain/MobileNetV1_pretrained ^
--checkpoint_dir ./output/mobilenetv1 --num_epochs 1 --batch_size 32 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof


:st_pact_quant_aware
call :pact_quant_aware_not_usepact
call :pact_quant_aware_usepact
call :pact_quant_aware_load
goto :eof

:st_pact_quant_aware
cd %repo_path%/PaddleSlim/demo/quant/pact_quant_aware
set model=st_pact_quant_aware_not_usepact
cd %repo_path%/PaddleSlim/demo/quant/pact_quant_aware
python train.py --model MobileNetV3_large_x1_0 ^
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained ^
--num_epochs 1 --lr 0.0001 --use_pact False --batch_size 32 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:pact_quant_aware_usepact
cd %repo_path%/PaddleSlim/demo/quant/pact_quant_aware
set model=st_pact_quant_aware_usepact
python train.py --model MobileNetV3_large_x1_0 ^
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained ^
--num_epochs 1 --lr 0.0001 --use_pact True --batch_size 16 --lr_strategy=piecewise_decay ^
--step_epochs 2 --l2_decay 1e-5 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

::load
:pact_quant_aware_load
cd %repo_path%/PaddleSlim/demo/quant/pact_quant_aware
set model=st_pact_quant_aware_load
python train.py --model MobileNetV3_large_x1_0 ^
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained ^
--num_epochs 2 --lr 0.0001 --use_pact True --batch_size 16 --lr_strategy=piecewise_decay ^
--step_epochs 20 --l2_decay 1e-5 ^
--checkpoint_dir ./output/MobileNetV3_large_x1_0/0 ^
--checkpoint_epoch 0 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:dy_quant_v1
cd %repo_path%/PaddleSlim/demo/dygraph/quant
set model=dy_quant_v1
python train.py --model=mobilenet_v1 ^
--pretrained_model '../../pretrain/MobileNetV1_pretrained' ^
--num_epochs 1 --batch_size 32 > %log_path%/%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:all_prune
call :st_prune_fpgm_MobileNetV2
call :st_prune_fpgm_resnet34_50
call :st_prune_fpgm_resnet34_42

call :unstructured_prune
goto :eof

:st_prune_fpgm_MobileNetV2
cd %repo_path%/PaddleSlim/demo/prune
set model=st_prune_fpgm_MobileNetV2
python train.py --model="MobileNetV2" --pretrained_model="../pretrain/MobileNetV2_pretrained" ^
    --data="imagenet" --pruned_ratio=0.325 --lr=0.001 --num_epochs=2 --test_period=1 ^
    --step_epochs 30 60 80 --l2_decay=1e-4 --lr_strategy="piecewise_decay" ^
    --criterion="geometry_median" --model_path="./output/fpgm_mobilenetv2_models" ^
    --save_inference True --batch_size 64 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:st_prune_fpgm_resnet34_50
set model=st_prune_fpgm_resnet34_50
cd %repo_path%/PaddleSlim/demo/prune
python train.py --model="ResNet34" --pretrained_model="../pretrain/ResNet34_pretrained" ^
    --data="imagenet" --pruned_ratio=0.3125 --lr=0.001 --num_epochs=2 --test_period=1 ^
    --step_epochs 30 60 --l2_decay=1e-4 --lr_strategy="piecewise_decay" --criterion="geometry_median" ^
    --model_path="./output/fpgm_resnet34_50_models" --batch_size 128 ^
    --save_inference True >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
set model=slim_prune_fpgm_resnet34_50_eval
python eval.py --model "ResNet34" --data "imagenet" ^
    --model_path "./output/fpgm_resnet34_50_models/0" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:st_prune_fpgm_resnet34_42
cd %repo_path%/PaddleSlim/demo/prune
set model=st_prune_fpgm_resnet34_42
python train.py --model="ResNet34" --pretrained_model="../pretrain/ResNet34_pretrained" ^
    --data="imagenet" --pruned_ratio=0.25  --num_epochs=2 --test_period=1 ^
    --lr_strategy="piecewise_decay" --batch_size 128 ^
    --criterion="geometry_median" --model_path="./output/fpgm_resnet34_025_120_models" ^
    --save_inference True > %log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
set model=st_slim_prune_fpgm_resnet34_42_eval
python eval.py --model "ResNet34" --data "imagenet" ^
--model_path "./output/fpgm_resnet34_025_120_models/0" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:st_unstructured_prune_ratio

cd %repo_path%/PaddleSlim/demo/unstructured_prune
set model=st_unstructured_prune_ratio
python train.py --batch_size 64 --pretrained_model ../pretrain/MobileNetV1_pretrained ^
    --lr 0.05 --pruning_mode ratio --ratio 0.55 --data imagenet ^
    --lr_strategy piecewise_decay --step_epochs 1 2 3 --num_epochs 1 --test_period 1 ^
    --model_path st_ratio_models >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

echo run st_unstructured_prune_threshold_mnist
set model=st_unstructured_prune_threshold
python train.py --batch_size 256 --pretrained_model ../pretrain/MobileNetV1_pretrained ^
    --lr 0.05 --pruning_mode threshold --threshold 0.01 --data mnist ^
    --lr_strategy piecewise_decay --step_epochs 1 2 3 --num_epochs 1 --test_period 1 ^
    --model_path st_unstructured_models_mnist >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof


:dy_unstructured_prune_threshold

cd %repo_path%/PaddleSlim/demo/dygraph/unstructured_pruning
set model=dy_threshold_prune_T
python -m paddle.distributed.launch ^
--log_dir train_dy_threshold_prune_log train.py ^
--data imagenet ^
--lr 0.05 ^
--pruning_mode threshold ^
--threshold 0.01 ^
--batch_size 64 ^
--lr_strategy piecewise_decay ^
--step_epochs 1 2 3 ^
--num_epochs 5 --model_period 2 ^
--test_period 1 ^
--model_path "./dy_threshold_models" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

set model=dy_threshold_prune_eval
python evaluate.py --pruned_model "./dy_threshold_models/model.pdparams" ^
--data imagenet >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

set model=dy_threshold_prune_T_load
python -m paddle.distributed.launch ^
--log_dir train_dy_ratio_log train.py ^
--data imagenet ^
--lr 0.05 ^
--pruning_mode threshold ^
--threshold 0.01 ^
--batch_size 64 ^
--lr_strategy piecewise_decay ^
--step_epochs 1 2 3 ^
--num_epochs 3 ^
--test_period 1 ^
--model_path "./dy_threshold_models" ^
--pretrained_model "./dy_threshold_models/model.pdparams" ^
--last_epoch 1 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

set model=dy_threshold_prune_cifar10_T
python train.py --data cifar10 --lr 0.05 ^
--pruning_mode threshold --num_epochs 30 ^
--threshold 0.01 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:printInfo
if %1 == 1 (
    move %log_path%\%model%.log %log_path%\FAIL_%model%.log
    echo  FAIL_%model%.log
    type %log_path%\FAIL_%model%.log
    echo  FAIL_%model%.log >> %log_path%\result.log
) else (
    move %log_path%\%model%.log %log_path%\SUCCESS_%model%.log
    echo SUCCESS_%model%.log
    echo SUCCESS_%model%.log >> %log_path%\result.log
)
goto :eof
