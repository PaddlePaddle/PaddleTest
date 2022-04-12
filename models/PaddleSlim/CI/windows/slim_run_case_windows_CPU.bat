cd %repo_path%/PaddleSlim/demo
mkdir data && cd data
mklink /J ILSVRC2012 %data_path%\ILSVRC2012_data_demo\ILSVRC2012
cd ../
mklink /J pretrain %pretrain_path%

echo -----------------run P0 CPU case:start ------------------------

echo 1 distillion
call :all_distillation

echo 2 quant
call :all_quant

echo 3 prune
call :all_prune

echo 4 nas
call :all_nas

echo -----------------run P0 CPU case:end -------------------------

:all_distillation
call :dist_res50_v1
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


:all_quant
call :quant_aware_v1_MobileNet
call :st_quant_post_v1_hist
call :pact_quant_aware_usepact
call :dy_pact_quant_v3
goto :eof


:quant_aware_v1_MobileNet
cd %repo_path%/PaddleSlim/demo/quant/quant_aware
set model=st_quant_aware_v1_MobileNet
python train.py --model ResNet34 ^
--pretrained_model ../../pretrain/ResNet34_pretrained --use_gpu=False ^
--checkpoint_dir ./output/ResNet34 --num_epochs 1 --batch_size 32 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

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

:st_quant_post_v1_hist
cd %repo_path%/PaddleSlim/demo/quant/quant_post
set model=st_quant_post_v1_T_hist

wget -P inference_model https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar --no-check-certificate
cd inference_model/
tar -xf MobileNetV1_infer.tar
cd ..

python quant_post.py --model_path ./inference_model/MobileNetV1_infer/ ^
--save_path ./quant_model/hist_bc/MobileNetV1 --use_gpu False ^
--bias_correction True >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:pact_quant_aware_usepact
cd %repo_path%/PaddleSlim/demo/quant/pact_quant_aware
set model=st_pact_quant_aware_usepact
python train.py --model MobileNetV3_large_x1_0 --use_gpu False ^
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained ^
--num_epochs 1 --lr 0.0001 --use_pact True --batch_size 16 --lr_strategy=piecewise_decay ^
--step_epochs 2 --l2_decay 1e-5 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:dy_pact_quant_v3
set model=dy_pact_quant_v3
python train.py  --lr=0.001 --batch_size 32 --use_pact=True --num_epochs=2 --l2_decay=2e-5 ^
    --use_gpu False ^
    --ls_epsilon=0.1  --pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained  ^
    > %log_path%/%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:all_prune
call :st_prune_fpgm_MobileNetV1
call :st_prune_ResNet50
call :dy_prune_ResNet34_f42
call :st_unstructured_prune_threshold
call :dy_unstructured_prune_ratio
goto :eof

:st_prune_fpgm_MobileNetV1
cd %repo_path%/PaddleSlim/demo/prune
set model=st_prune_fpgm_MobileNetV1
python train.py --model="MobileNet" --pretrained_model="../pretrain/MobileNetV1_pretrained" ^
    --data="imagenet" --pruned_ratio=0.3125 --lr=0.1 --num_epochs=1 --test_period=1 ^
    --step_epochs 30 60 90 --batch_size 64 ^
    --l2_decay=3e-5 --lr_strategy="piecewise_decay" ^ --criterion="geometry_median" ^
    --use_gpu False ^
    --model_path="./fpgm_mobilenetv1_models" --save_inference True  >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:st_prune_ResNet50
set model=st_prune_ResNet50
cd %repo_path%/PaddleSlim/demo/prune
python train.py --model ResNet50 --pruned_ratio 0.31 --data "imagenet" ^
    --save_inference True --pretrained_model ../pretrain/ResNet50_pretrained ^
    --use_gpu False ^
    --num_epochs 1 --batch_size 64 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:dy_prune_ResNet34_f42
::train--恢复train--eval--export
set model=dy_prune_ResNet34_f42_gpu1
cd %repo_path%/PaddleSlim/demo/dygraph/pruning
mkdir data && cd data
mklink /J ILSVRC2012 %data_path%\ILSVRC2012_data_demo\ILSVRC2012
cd ..
python train.py --use_gpu False ^ --model="resnet34" --data="imagenet" ^
    --pruned_ratio=0.25 --num_epochs=1 --batch_size=128 --lr_strategy="cosine_decay" ^
    --criterion="fpgm" ^
    --model_path="./fpgm_resnet34_025_120_models" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
set model=dy_prune_ResNet50_f42_gpu1_load
python train.py --use_gpu False ^ --model="resnet34" --data="imagenet" --pruned_ratio=0.25 ^
    --num_epochs=2 --batch_size=128 --lr_strategy="cosine_decay" --criterion="fpgm" ^
    --model_path="./fpgm_resnet34_025_120_models" ^
    --checkpoint="./fpgm_resnet34_025_120_models/0" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

set model=dy_prune_ResNet50_f42_gpu1_eval
python eval.py --checkpoint=./fpgm_resnet34_025_120_models/1 --model="resnet34" ^
    --use_gpu False ^
    --pruned_ratio=0.25 --batch_size=128 >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

set model=dy_prune_ResNet50_f42_gpu1_export
python export_model.py --checkpoint=./fpgm_resnet34_025_120_models/final ^
    --model="resnet34" --pruned_ratio=0.25 --output_path=./infer_final/resnet >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof

:st_unstructured_prune_threshold
set model=st_unstructured_prune_threshold
cd %repo_path%/PaddleSlim/demo/unstructured_prune
python train.py --batch_size 64 --pretrained_model ../pretrain/MobileNetV1_pretrained ^
    --lr 0.05 --pruning_mode threshold --threshold 0.01 --data imagenet ^
    --lr_strategy piecewise_decay --step_epochs 1 2 3 --num_epochs 5 --model_period 2 ^
    --test_period 1 ^
    --use_gpu=False ^
    --model_path ./st_unstructured_models >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%

:dy_unstructured_prune_ratio
echo run dy_unstructured_prune_ratio
cd %repo_path%/PaddleSlim/demo/dygraph/unstructured_pruning
python -m paddle.distributed.launch ^
--log_dir dy_ratio_prune_ratio_log train.py ^
--data imagenet --lr 0.05 --pruning_mode ratio --ratio 0.55 ^
--batch_size 64 --lr_strategy piecewise_decay ^
--step_epochs 1 2 3 --num_epochs 5 --model_period 2 ^
--use_gpu=False ^
--test_period 1 --model_path "./dy_threshold_models" >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%


:all_nas
call :sa_nas_v2
goto :eof

:sa_nas_v2
cd %repo_path%/PaddleSlim/demo/nas
set model=sa_nas_v2
python sa_nas_mobilenetv2.py --search_steps 1 --retain_epoch 1 ^
--use_gpu False >%log_path%\%model%.log 2>&1
call :printInfo  %errorlevel%
goto :eof


:printInfo
if %1 == 1 (
    move %log_path%\%model%.log %log_path%\FAIL_%model%.log
    echo  FAIL_%model%.log
    echo  FAIL_%model%.log >> %log_path%\result.log
) else (
    move %log_path%\%model%.log %log_path%\SUCCESS_%model%.log
    echo SUCCESS_%model%.log
    echo SUCCESS_%model%.log >> %log_path%\result.log
)
goto :eof
