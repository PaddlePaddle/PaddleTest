#!/usr/bin/env bash
set -e

export fleetx_path=/paddle/PaddleFleetX
export data_path=/fleetx_data
export log_path=/paddle/log_fleetx

fleet_gpu_model_list=( \
    gpt_preprocess_data \
    gpt_345M_single \
    gpt_1.3B_dp \
    gpt_6.7B_stage2_dp2_sharding4 \
    gpt_6.7B_stage3_dp2_sharding4 \
    gpt_6.7B_stage2_sharding8 \
    gpt_175B_DP1_MP4_PP2 \
    gpt_175B_DP1_MP4_PP2_sp \
    gpt_175B_DP1_MP8_PP1 \
    gpt_175B_DP1_MP8_PP1_sp \
    gpt_175B_DP1_MP1_PP8 \
    gpt_345M_mp8_qat \
    gpt_generation_345M_single \
    gpt_generation_345M_hybrid  \
    gpt_inference_345M_single \
    gpt_inference_345M_dp8 \
    gpt_345M_single_finetune \
    gpt_eval_WikiText \
    gpt_eval_LAMBADA \
    ernie_base_3D \
    ernie_dp2 \
    vit_cifar10_finetune \
    vit_qat \
    vit_inference \
    imagen_text2im_397M_64x64_single \
    imagen_text2im_397M_64x64_dp8 \
    imagen_text2im_2B_64x64_sharding8 \
    imagen_text2im_64x64_DebertaV2_dp8 \
    imagen_super_resolution_256_single_card \
    imagen_super_resolution_256_dp8 \
    )


function gpt_preprocess_data() {
    cd ${fleetx_path}
    rm -rf log
    python ppfleetx/data/data_tools/gpt/raw_trans_to_json.py  \
        --input_path ./dataset/wikitext_103_en \
        --output_path ./dataset/wikitext_103_en/wikitext_103_en
    python ppfleetx/data/data_tools/gpt/preprocess_data.py \
        --model_name gpt2 \
        --tokenizer_name GPTTokenizer \
        --data_format JSON \
        --input_path ./dataset/wikitext_103_en/wikitext_103_en.jsonl \
        --append_eos \
        --output_prefix ./dataset/wikitext_103_en/wikitext_103_en  \
        --workers 40 \
        --log_interval 1000
    check_result $FUNCNAME
}

function gpt_345M_single() {
    cd ${fleetx_path}
    rm -rf log
    python tools/train.py \
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10
    check_result $FUNCNAME
}

function gpt_1.3B_dp() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10
    check_result $FUNCNAME
}

function gpt_6.7B_stage2_dp2_sharding4() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Distributed.sharding.sharding_degree=4 -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.sharding.reduce_overlap=False -o Distributed.sharding.broadcast_overlap=False \
        -o Engine.logging_freq=5
    check_result $FUNCNAME
}

function gpt_6.7B_stage3_dp2_sharding4() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Distributed.sharding.sharding_degree=4 -o Distributed.sharding.sharding_stage=3 \
        -o Distributed.sharding.reduce_overlap=False -o Distributed.sharding.broadcast_overlap=False \
        -o Engine.logging_freq=5
    check_result $FUNCNAME
}

function gpt_6.7B_stage2_sharding8() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=20 -o Engine.eval_freq=20 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Distributed.sharding.sharding_degree=8 -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.sharding.reduce_overlap=True -o Distributed.sharding.broadcast_overlap=True \
        -o Engine.logging_freq=5
    check_result $FUNCNAME
}

function gpt_175B_DP1_MP4_PP2() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=4 -o Distributed.pp_degree=2 \
        -o Model.sequence_parallel=False
    check_result $FUNCNAME
}

function gpt_175B_DP1_MP4_PP2_sp() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=4 -o Distributed.pp_degree=2 -o Model.sequence_parallel=True
    check_result $FUNCNAME
}

function gpt_175B_DP1_MP8_PP1() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=16 -o Model.num_attention_heads=16 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=8 -o Distributed.pp_degree=1 \
        -o Model.sequence_parallel=False
    check_result $FUNCNAME
}

function gpt_175B_DP1_MP8_PP1_sp() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=16 -o Model.num_attention_heads=16 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=8 -o Distributed.pp_degree=1 -o Model.sequence_parallel=True
    check_result $FUNCNAME
}

function gpt_175B_DP1_MP1_PP8() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=32 -o Model.num_attention_heads=16 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=1 \
        -o Distributed.mp_degree=1 -o Distributed.pp_degree=8 \
        -o Model.virtual_pp_degree=2 -o Distributed.pp_recompute_interval=2 \
        -o Model.fused_linear=True -o Model.use_recompute=True \
        -o Model.sequence_parallel=False
    check_result $FUNCNAME
}

function gpt_345M_mp8_qat() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/qat_gpt_345M_mp8.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=8 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10
    check_result $FUNCNAME
}

function gpt_generation_345M_single() {
    cd ${fleetx_path}
    rm -rf log
    python tasks/gpt/generation.py \
        -c ppfleetx/configs/nlp/gpt/generation_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/
    check_result $FUNCNAME
    # check_generation_txt $FUNCNAME ./log
}

function gpt_generation_345M_hybrid() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0" tasks/gpt/generation.py \
        -c ppfleetx/configs/nlp/gpt/generation_gpt_345M_dp8.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/
    check_result $FUNCNAME
    # tail -12 log/workerlog.0 > ./generation_345M_dp8.txt
    # check_generation_txt $FUNCNAME ./generation_345M_dp8.txt
}

function gpt_inference_345M_single() {
    cd ${fleetx_path}
    rm -rf log
    python tools/export.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/
    python tasks/gpt/inference.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml
    check_result $FUNCNAME
    # check_generation_txt $FUNCNAME ./log
}

function gpt_inference_345M_dp8() {
    cd ${fleetx_path}
    rm -rf log
    python -m paddle.distributed.launch --devices "0" tools/export.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/
    python -m paddle.distributed.launch --devices "0" \
        tasks/gpt/inference.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml
    check_result $FUNCNAME
    # tail -12 log/workerlog.0 > ./inference_345M_single.txt
    # check_generation_txt $FUNCNAME ./inference_345M_single.txt
}

function gpt_345M_single_finetune() {
    cd ${fleetx_path}
    rm -rf log
    python ./tools/train.py \
        -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
        -o Engine.num_train_epochs=1 \
        -o Data.Train.dataset.name=WNLI \
        -o Data.Train.dataset.root=./dataset/WNLI/ \
        -o Data.Eval.dataset.name=WNLI \
        -o Data.Eval.dataset.root=./dataset/WNLI/ \
        -o Data.Eval.dataset.split=dev \
        -o Model.num_classes=2
    check_result $FUNCNAME
}

function gpt_eval_WikiText() {
    cd ${fleetx_path}
    rm -rf log
    python ./tools/eval.py \
        -c ./ppfleetx/configs/nlp/gpt/eval_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826 \
        -o Offline_Eval.eval_path=./wikitext-103/wiki.valid.tokens \
        -o Offline_Eval.overlapping_eval=32 \
        -o Offline_Eval.batch_size=16 \
        -o Engine.max_steps=20
    check_result $FUNCNAME
}

function gpt_eval_LAMBADA() {
    cd ${fleetx_path}
    rm -rf log
    python ./tools/eval.py \
        -c ./ppfleetx/configs/nlp/gpt/eval_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826 \
        -o Offline_Eval.eval_path=./lambada_test.jsonl \
        -o Offline_Eval.cloze_eval=True \
        -o Offline_Eval.batch_size=16 \
        -o Engine.max_steps=20
    check_result $FUNCNAME
}

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
    loss=`tail log/workerlog.0 | grep 19/24 | cut -d " " -f14 `
    top1=`tail log/workerlog.0 | grep top1 |cut -d " " -f14 `
    if [[ ${AGILE_COMPILE_BRANCH} =~ "develop" ]];then
        check_diff 3.744531250 ${loss%?} ${FUNCNAME}_loss
        check_diff 0.217041 ${top1%?} ${FUNCNAME}_top1
    else
        check_diff 3.744726562 ${loss%?} ${FUNCNAME}_loss
        check_diff 0.216858 ${top1%?} ${FUNCNAME}_top1
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
    loss=`tail log/workerlog.0 | grep "eval" | cut -d " " -f11 `
    if [[ ${AGILE_COMPILE_BRANCH} =~ "develop" ]];then
        check_diff 2.299847364 ${loss%?} ${FUNCNAME}_loss
    else
        check_diff 2.299857140 ${loss%?} ${FUNCNAME}_loss
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
        # sleep 1
        echo "=========== ${model} run  end ==========="
      done
}


main() {
    run_gpu_models
}

main$@
