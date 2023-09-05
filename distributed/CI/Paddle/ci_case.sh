#!/usr/bin/env bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

export nlp_dir=/workspace/PaddleNLP/
export log_path=/workspace/PaddleNLP/model_logs
export case_path=/workspace/PaddleNLP/model_zoo/gpt-3
export data_path=/fleetx_data

unset CUDA_VISIBLE_DEVICES

function case_list_auto() {
    gpt_auto_recompute_bs16_fp32_DP1-MP1-PP1
    gpt_auto_recompute_bs16_fp16_o2_DP1-MP1-PP8
    gpt_auto_recompute_bs16_fp16_o2_DP1-MP2-PP4
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2
    gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage1
    gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage2
    gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage3
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage1
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage2
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage3
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage1
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage2
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage3
}

############ case start ############

function gpt_auto_recompute_bs16_fp32_DP1-MP1-PP1() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=$log_dir --devices=0 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=16 \
        -o Global.micro_batch_size=16 \
        -o Distributed.dp_degree=1 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=False \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.507633400
    ips_base=3504
    mem_base=11750.6
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP1-MP1-PP8() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=16 \
        -o Global.micro_batch_size=2 \
        -o Distributed.dp_degree=1 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=8 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.7 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.570028400
    ips_base=34566
    mem_base=2052.9
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP1-MP2-PP4() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=16 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=1 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=4 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.7 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.697089195
    ips_base=31923
    mem_base=1535.7
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.669981194
    ips_base=18986
    mem_base=2135.7
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage1() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=4 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=4 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.715782070
    ips_base=15418
    mem_base=1999.2
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage2() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=4 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=4 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.715772343
    ips_base=15639
    mem_base=1999.2
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage3() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=4 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=4 \
        -o Distributed.sharding.sharding_stage=3 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.681084633
    ips_base=14023
    mem_base=1747.6
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage1() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=2 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=4 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.3 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.579057693
    ips_base=19860
    mem_base=1709.8
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage2() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=2 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=4 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.3 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.579057693
    ips_base=20167
    mem_base=1709.8
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage3() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=2 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=4 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=3 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.3 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.585316849
    ips_base=15833
    mem_base=1591.6
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage1() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.669967556
    ips_base=19608
    mem_base=1384.7
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage2() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.669967556
    ips_base=19818
    mem_base=1384.7
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage3() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=3 \
        -o Distributed.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.694304180
    ips_base=16810
    mem_base=1288.5
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="   
}
############ case end ############

function before_hook() {
    # env FLAGS
    export FLAGS_new_executor_micro_batching=True  # True：打开新执行器
    export FLAGS_embedding_deterministic=1         # 1：关闭随机性
    export FLAGS_cudnn_deterministic=1             # 1：关闭随机性
    unset CUDA_MODULE_LOADING
    env | grep FLAGS
    # requirements
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    python -m pip install -r requirements.txt --force-reinstall

    rm -rf data
    if [[ -e ${data_path}/data ]]; then
        echo "data downloaded"
    else
        # download data for gpt
        mkdir ${data_path}/data;
        wget -O ${data_path}/data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy;
        wget -O ${data_path}/data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz;
    fi

    cp -r ${data_path}/data ${case_path}/
}

function check_result() {
    if [ $? -ne 0 ];then
        mv ${log_path}/$1 ${log_path}/$1_FAIL.log
        echo -e "\033 ${log_path}/$1_FAIL \033"
        cat ${log_path}/$1_FAIL.log
        exit -1
    fi

    if [ $# -ne 7 ]; then
        echo -e "\033 parameter transfer failed: $@ \033" 
        cat ${log_path}/$1_FAIL.log
        exit -1
    fi

    echo -e "loss_base: $2 loss_test: $3" | tee -a ${log_path}/$1
    if [ $2 != $3 ];then
        mv ${log_path}/$1 ${log_path}/$1_FAIL.log
        echo -e "\033 ${log_path}/$1 loss diff check failed! \033"
        cat ${log_path}/$1_FAIL.log
        exit -1
    else
        echo -e "\033 $1 loss diff check successfully! \033" | tee -a $log_path/result.log
    fi

    diff=$(echo $4 $5|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
    echo -e "ips_base: $4 ips_test: $5 ips_diff: $diff% " | tee -a $log_path/result.log
    v1=$(echo $diff 5.0|awk '{print($1>=$2)?"0":"1"}')
    v2=$(echo $diff -5.0|awk '{print($1<=$2)?"0":"1"}')
    if [[ $v1 == 0 ]];then
      echo -e "\033 $1 IPS increase greater than 5%! \033" | tee -a $log_path/result.log
    fi
    if [[ $v2 == 0 ]];then
      echo -e "\033 $1 ips diff check failed! \033" | tee -a $log_path/result.log
      exit -1
    fi

    echo -e "mem_base: $6 mem_test: $7" | tee -a $log_path/result.log
    if [ $6 != $7 ];then
      echo -e "\033 $1 mem diff check failed! \033" | tee -a $log_path/result.log
      exit -1
    fi
}

main() {
    cd ${case_path}
    before_hook
    case_list_auto
}

main$@
