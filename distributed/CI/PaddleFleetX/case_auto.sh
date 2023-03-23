#!/usr/bin/env bash
set -e

export fleetx_path=/paddle/PaddleFleetX
export log_path=/paddle/log_fleetx

fleet_gpu_model_list=( \
    gpt_save_ckpt \
    gpt_auto_serial \
    gpt_auto_dp2mp2 \
    gpt_auto_dp2pp2 \
    gpt_auto_mp2pp2 \
    gpt_auto_dp2mp2pp2 \
    gpt_auto_dp2sharding2 \
    gpt_auto_dp2mp2sharding2 \
    gpt_auto_dp2pp2sharding2 \
    gpt_auto_dp2mp2pp2sharding2 \
    gpt_auto_pass_o1_stage1 \
    gpt_auto_pass_o1_stage2 \
    gpt_auto_pass_o2_stage1 \
    gpt_auto_pass_o2_stage2 \
    gpt_auto_pass_o3_stage1 \
    gpt_auto_pass_o3_stage2 \
    gpt_auto_dp2mp2pp2_o2 \
    gpt_auto_export \
    )


function gpt_save_ckpt() {
    cd ${fleetx_path}
    rm -rf log
    python ./tools/train.py \
        -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=False \
        -o Model.hidden_dropout_prob=0. \
        -o Model.attention_probs_dropout_prob=0. \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Global.local_batch_size=2 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=1 \
        -o Engine.save_load.save_steps=1 \
        -o Engine.save_load.output_dir="./ckpt_dynamic"
    check_result $FUNCNAME
}

function gpt_auto_serial() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=False \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=2 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=1 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Engine.save_load.ckpt_dir="./ckpt_dynamic/epoch_0_step_1/auto_infer/auto"
    check_result $FUNCNAME
    loss=`tail -5 $log_dir/workerlog.0 | grep "lr:" | cut -d " " -f5 `
    check_diff 10.9509 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_dp2mp2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=False \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=2 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Engine.save_load.ckpt_dir="./ckpt_dynamic/epoch_0_step_1/auto_infer/auto"
    check_result $FUNCNAME
    loss=`tail -5 $log_dir/workerlog.0 | grep "lr:" | cut -d " " -f5 `
    check_diff 10.9697 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_mp2pp2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=False \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=2 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=1 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Engine.save_load.ckpt_dir="./ckpt_dynamic/epoch_0_step_1/auto_infer/auto"
    check_result $FUNCNAME
    loss=`tail -5 $log_dir/workerlog.2 | grep "lr:" | cut -d " " -f5 `
    check_diff 10.9509 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_dp2pp2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=False \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=2 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Engine.save_load.ckpt_dir="./ckpt_dynamic/epoch_0_step_1/auto_infer/auto"
    check_result $FUNCNAME
    loss1=`tail -5 $log_dir/workerlog.2 | grep "lr:" | cut -d " " -f5 `
    loss2=`tail -5 $log_dir/workerlog.3 | grep "lr:" | cut -d " " -f5 `
    loss=$(echo $loss1 $loss2 | awk '{printf("%.4f",($1+$2)/2)}')
    check_diff 10.9732 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_dp2mp2pp2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=False \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=2 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Engine.save_load.ckpt_dir="./ckpt_dynamic/epoch_0_step_1/auto_infer/auto"
    check_result $FUNCNAME
    loss1=`tail -5 $log_dir/workerlog.4 | grep "lr:" | cut -d " " -f5 `
    loss2=`tail -5 $log_dir/workerlog.6 | grep "lr:" | cut -d " " -f5 `
    loss=$(echo $loss1 $loss2 | awk '{printf("%.4f",($1+$2)/2)}')
    check_diff 10.9732 ${loss} ${FUNCNAME}_loss
}
function gpt_auto_dp2mp2pp2_o2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=$log_dir --devices="0,1,2,3,4,5,6,7" \
        tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp8.yaml \
        -o Engine.mix_precision.enable=True
        -o Engine.mix_precision.level="o2" \
        -o Model.hidden_size=1024 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Engine.verbose=3 \
        -o Model.type_vocab_size=1
    check_result $FUNCNAME
}

function gpt_auto_dp2sharding2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=False \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=2 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Engine.save_load.ckpt_dir="./ckpt_dynamic/epoch_0_step_1/auto_infer/auto"
    check_result $FUNCNAME
    loss=`tail -5 $log_dir/workerlog.0 | grep "lr:" | cut -d " " -f5 `
    check_diff 10.9697 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_dp2mp2sharding2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=False \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=2 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Engine.save_load.ckpt_dir="./ckpt_dynamic/epoch_0_step_1/auto_infer/auto"
    check_result $FUNCNAME
    loss=`tail -5 $log_dir/workerlog.0 | grep "lr:" | cut -d " " -f5 `
    check_diff 10.9697 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_dp2pp2sharding2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=False \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=2 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Engine.save_load.ckpt_dir="./ckpt_dynamic/epoch_0_step_1/auto_infer/auto"
    check_result $FUNCNAME
    loss1=`tail -5 $log_dir/workerlog.2 | grep "lr:" | cut -d " " -f5 `
    loss2=`tail -5 $log_dir/workerlog.3 | grep "lr:" | cut -d " " -f5 `
    loss=$(echo $loss1 $loss2 | awk '{printf("%.4f",($1+$2)/2)}')
    check_diff 10.9732 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_dp2mp2pp2sharding2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=False \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=2 \
        -o Global.micro_batch_size=2 \
        -o Engine.max_steps=10 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Engine.save_load.ckpt_dir="./ckpt_dynamic/epoch_0_step_1/auto_infer/auto"
    check_result $FUNCNAME
    loss1=`tail -5 $log_dir/workerlog.4 | grep "lr:" | cut -d " " -f5 `
    loss2=`tail -5 $log_dir/workerlog.6 | grep "lr:" | cut -d " " -f5 `
    loss=$(echo $loss1 $loss2 | awk '{printf("%.4f",($1+$2)/2)}')
    check_diff 10.9732 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_pass_o1_stage1() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=True
        -o Engine.mix_precision.level="o1" \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=1 \
        -o Engine.max_steps=4 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=1

    check_result $FUNCNAME
    loss1=`tail -5 $log_dir/workerlog.4 | grep "lr:" | cut -d " " -f5 `
    loss2=`tail -5 $log_dir/workerlog.6 | grep "lr:" | cut -d " " -f5 `
    loss=$(echo $loss1 $loss2 | awk '{printf("%.4f",($1+$2)/2)}')
    check_diff 11.0779 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_pass_o1_stage2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=True
        -o Engine.mix_precision.level="o1" \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=1 \
        -o Engine.max_steps=4 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2
    check_result $FUNCNAME
    loss1=`tail -5 $log_dir/workerlog.4 | grep "lr:" | cut -d " " -f5 `
    loss2=`tail -5 $log_dir/workerlog.6 | grep "lr:" | cut -d " " -f5 `
    loss=$(echo $loss1 $loss2 | awk '{printf("%.4f",($1+$2)/2)}')
    check_diff 11.0779 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_pass_o2_stage1() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=True
        -o Engine.mix_precision.level="o2" \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=1 \
        -o Engine.max_steps=4 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=1
    check_result $FUNCNAME
    loss1=`tail -5 $log_dir/workerlog.4 | grep "lr:" | cut -d " " -f5 `
    loss2=`tail -5 $log_dir/workerlog.6 | grep "lr:" | cut -d " " -f5 `
    loss=$(echo $loss1 $loss2 | awk '{printf("%.4f",($1+$2)/2)}')
    check_diff 11.0779 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_pass_o2_stage2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=True
        -o Engine.mix_precision.level="o2" \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=1 \
        -o Engine.max_steps=4 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2
    check_result $FUNCNAME
    loss1=`tail -5 $log_dir/workerlog.4 | grep "lr:" | cut -d " " -f5 `
    loss2=`tail -5 $log_dir/workerlog.6 | grep "lr:" | cut -d " " -f5 `
    loss=$(echo $loss1 $loss2 | awk '{printf("%.4f",($1+$2)/2)}')
    check_diff 11.0779 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_pass_o3_stage1() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=True
        -o Engine.mix_precision.level="o3" \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=1 \
        -o Engine.max_steps=4 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=1
    check_result $FUNCNAME
    loss1=`tail -5 $log_dir/workerlog.4 | grep "lr:" | cut -d " " -f5 `
    loss2=`tail -5 $log_dir/workerlog.6 | grep "lr:" | cut -d " " -f5 `
    loss=$(echo $loss1 $loss2 | awk '{printf("%.4f",($1+$2)/2)}')
    check_diff 11.0779 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_pass_o3_stage2() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
        ./tools/auto.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Engine.mix_precision.enable=True
        -o Engine.mix_precision.level="o3" \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4 \
        -o Model.use_recompute=True \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=1 \
        -o Engine.max_steps=4 \
        -o Engine.logging_freq=1 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2
    check_result $FUNCNAME
    loss1=`tail -5 $log_dir/workerlog.4 | grep "lr:" | cut -d " " -f5 `
    loss2=`tail -5 $log_dir/workerlog.6 | grep "lr:" | cut -d " " -f5 `
    loss=$(echo $loss1 $loss2 | awk '{printf("%.4f",($1+$2)/2)}')
    check_diff 11.0779 ${loss} ${FUNCNAME}_loss
}

function gpt_auto_export() {
    cd ${fleetx_path}
    log_dir=log_auto
    rm -rf $log_dir

    python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1" \
        ./tools/auto_export.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/generation_gpt_345M_mp2.yaml \
        -o Model.num_layers=4 \
        -o Model.num_attention_heads=4
    check_result $FUNCNAME
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
