#!/usr/bin/env bash
set -e

export fleetx_path=/paddle/PaddleFleetX
export data_path=/fleetx_data
export log_path=/paddle/log_fleetx

fleet_gpu_model_list=( \
    ernie_base_3D \
    ernie_dp2 \
    vit_cifar10_finetune \
    vit_qat \
    vit_inference \
    imagen_text2im_397M_64x64_single \
    imagen_text2im_397M_64x64_dp8 \
    imagen_text2im_64x64_DebertaV2_dp8 \
    imagen_text2im_2B_64x64_sharding8 \
    imagen_super_resolution_256_single_card \
    imagen_super_resolution_256_dp8 \
    )

function ernie_base_3D() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py \
        -c ppfleetx/configs/nlp/ernie/pretrain_ernie_base_3D.yaml \
        -o Data.Train.dataset.input_dir=./dataset/ernie \
        -o Data.Eval.dataset.input_dir=./dataset/ernie \
        -o Engine.max_steps=10
    check_result $FUNCNAME
}

function ernie_dp2() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1" tools/train.py \
        -c ppfleetx/configs/nlp/ernie/pretrain_ernie_base_3D.yaml \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=8 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=1 \
        -o Data.Train.dataset.input_dir=./dataset/ernie \
        -o Data.Eval.dataset.input_dir=./dataset/ernie \
        -o Engine.max_steps=10
    check_result $FUNCNAME
}

function vit_cifar10_finetune() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py \
        -c  ppfleetx/configs/vis/vit/ViT_tiny_patch16_224_ci_cifar10_1n8c_dp_fp16o2.yaml
    check_result $FUNCNAME
    loss=`cat log/workerlog.0 | grep 19/24 | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    top1=`cat log/workerlog.0 | grep top1 | awk -F 'top1 = ' '{print $2}' | awk -F ',' '{print $1}'`
    if [[ ${AGILE_COMPILE_BRANCH} =~ "develop" ]];then
        check_diff 3.567650485 ${loss} ${FUNCNAME}_loss
        check_diff 0.197876 ${top1} ${FUNCNAME}_top1
    else
        check_diff 3.744726562 ${loss} ${FUNCNAME}_loss
        check_diff 0.216858 ${top1} ${FUNCNAME}_top1
    fi
}

function vit_qat() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py \
        -c ppfleetx/configs/vis/vit/ViT_base_patch16_384_ft_qat_cifar10_1n8c_dp_fp16o2.yaml \
        -o Engine.run_mode='step' \
        -o Engine.num_train_epochs=1 \
        -o Engine.max_steps=100 \
        -o Engine.eval_freq=100 \
        -o Optimizer.lr.learning_rate=0.000025 \
        -o Data.Train.sampler.batch_size=32 \
        -o Engine.save_load.save_steps=100 \
        -o Model.model.pretrained.prefix_path=./ckpt/model
    check_result $FUNCNAME
    loss=`cat log/workerlog.0 | grep "1/20" | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}' `
    if [[ ${AGILE_COMPILE_BRANCH} =~ "develop" ]];then
        check_diff 2.299863338 ${loss} ${FUNCNAME}_loss
    else
        check_diff 2.299857140 ${loss} ${FUNCNAME}_loss
    fi
}

function vit_inference() {
    cd ${fleetx_path}
    rm -rf log
    python tools/export.py \
        -c ppfleetx/configs/vis/vit/ViT_base_patch16_224_inference.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/
    rm -rf shape.pbtxt
    if [[ ${AGILE_COMPILE_BRANCH} =~ "develop" ]];then
        python projects/vit/inference.py -c ppfleetx/configs/vis/vit/ViT_base_patch16_224_inference.yaml
    else
        python projects/vit/inference_vit.py -c ppfleetx/configs/vis/vit/ViT_base_patch16_224_inference.yaml
    fi
    check_result $FUNCNAME
}

function imagen_text2im_397M_64x64_single() {
    cd ${fleetx_path}
    rm -rf log
    python tools/train.py \
        -c ppfleetx/configs/multimodal/imagen/imagen_397M_text2im_64x64.yaml \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1
    check_result $FUNCNAME
}

function imagen_text2im_397M_64x64_dp8() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py \
        -c ppfleetx/configs/multimodal/imagen/imagen_397M_text2im_64x64.yaml \
        -o Distributed.dp_degree=8 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1
    check_result $FUNCNAME
}

function imagen_text2im_2B_64x64_sharding8() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py \
        -c ppfleetx/configs/multimodal/imagen/imagen_text2im_64x64_T5-11B.yaml \
        -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.sharding.sharding_degree=8 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1
    check_result $FUNCNAME
}

function imagen_text2im_64x64_DebertaV2_dp8() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py \
        -c ppfleetx/configs/multimodal/imagen/imagen_text2im_64x64_DebertaV2.yaml \
        -o Distributed.dp_degree=8 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1
    check_result $FUNCNAME
}

function imagen_super_resolution_256_single_card() {
    cd ${fleetx_path}
    rm -rf log
    python tools/train.py \
        -c ppfleetx/configs/multimodal/imagen/imagen_super_resolution_256.yaml \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1
    check_result $FUNCNAME
}

function imagen_super_resolution_256_dp8() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py \
        -c ./ppfleetx/configs/multimodal/imagen/imagen_super_resolution_256.yaml \
        -o Distributed.dp_degree=8 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1
    check_result $FUNCNAME
}

function check_generation_txt() {
    if [[ -e ./generation_base.txt ]]; then
        echo "check generation txt"
    else
        cp /paddle/PaddleTest/distributed/CI/PaddleFleetX/generation_base.txt ./
    fi
    diff generation_base.txt $2
    if [ $? -ne 0 ];then
      echo -e "\033 $1 generation check failed! \033" | tee -a $log_path/result.log
    else
      echo -e "\033 $1 generation check successfully! \033" | tee -a $log_path/result.log
    fi
}

function check_result() {
    if [ $? -ne 0 ];then
      echo -e "\033 $1 model runs failed! \033" | tee -a $log_path/result.log
    else
      echo -e "\033 $1 model runs successfully! \033" | tee -a $log_path/result.log
    fi
}


function check_diff() {
    echo "base:$1 test:$2"
    if [ $1 != $2 ];then
      echo -e "\033 $3 model_diff runs failed! \033" | tee -a $log_path/result.log
      exit -1
    else
      echo -e "\033 $3 model_diff runs successfully! \033" | tee -a $log_path/result.log
    fi
}

function run_cpu_models(){
      for model in ${fleet_cpu_model_list[@]}
      do
        echo "=========== ${model} run begin ==========="
        $model
        sleep 3
        echo "=========== ${model} run  end ==========="
      done
}

function run_gpu_models(){
    cd
      for model in ${fleet_gpu_model_list[@]}
      do
        echo "=========== ${model} run begin ==========="
        $model
        echo "=========== ${model} run  end ==========="
      done
}

main() {
    run_gpu_models
}


main$@
