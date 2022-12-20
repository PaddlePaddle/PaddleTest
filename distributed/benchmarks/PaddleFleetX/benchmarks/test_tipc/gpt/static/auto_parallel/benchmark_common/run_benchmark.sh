#!/usr/bin/env bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# Test training benchmark for a model.
# Usage：bash benchmark/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} ${use_sharding}
function _set_params(){
    model_item=${1:-"model_item"}   # (必选) 模型 item
    fp_item=${2:-"o1"}            # (必选) o1|o2|o3
    dp_degree=${3:-"1"}             # (必选) dp数据并行度
    mp_degree=${4:-"1"}             # (必选) mp数据并行度
    pp_degree=${5:-"1"}             # (必选) pp数据并行度
    micro_batch_size=${6:-"2"}      # (必选) micro_batch_size
    global_batch_size=${7:-"16"}    # （必选）global_batch_size
    run_mode=${8:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP2-MP8-PP2|DP1-MP8-PP4|DP4-MP8-PP1
    device_num=${9:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="PaddleFleetX"          # (必选) 模型套件的名字
    speed_unit="samples/s"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    sharding_degree=${10:-"1"}      # (可选)
    sharding_stage=${11:-"1"}       # (可选)sharding case
    hidden_size=${12:-"5120"}       # (可选) 模型大小
    num_workers=${13:-0}                 # (可选)
    max_iter=${14:-100}                      # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件；或使用max_epoch参数
    base_batch_size=$global_batch_size
    verbose=${15:-"3"}         # (可选)是否打印性能数据
    logging_freq=${16:-"1"} # (可选)loss打印频率

    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${global_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    #
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed

    OUTPUT_PATH=${run_log_path}/output
}

function _train(){
    batch_size=${local_batch_size}  # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs

    if [ -d $OUTPUT_PATH ]; then
        rm -rf $OUTPUT_PATH
    fi
    mkdir $OUTPUT_PATH

    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"

    if [ ${profiling} = "true" ];then
        add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
        log_file=${profiling_log_file}
    else
        add_options=""
        log_file=${train_log_file}
    fi

    local_batch_size=`expr ${global_batch_size} / ${dp_degree}`
    sharding_degree=${dp_degree}
    train_cmd="-o Model.hidden_dropout_prob=0 \
               -o Model.attention_probs_dropout_prob=0 \
               -o Model.use_recompute=True \
               -o Global.local_batch_size=${local_batch_size} \
               -o Global.micro_batch_size=${micro_batch_size} \
               -o Distributed.dp_degree=${dp_degree} \
               -o Distributed.mp_degree=${mp_degree} \
               -o Distributed.pp_degree=${pp_degree} \
               -o Distributed.sharding.sharding_degree=${sharding_degree} \
               -o Distributed.sharding.sharding_stage=${sharding_stage} \
               -o Engine.mix_precision.level=${fp_item} \
               -o Engine.max_steps=${max_iter} \
               -o Engine.eval_freq=100000 \
               -o Engine.verbose=${verbose} \
               -o Engine.logging_freq=${logging_freq} "

    if [ ${PADDLE_TRAINER_ID} ]
    then
        PADDLE_RANK_OPTION=" --rank ${PADDLE_TRAINER_ID}"
    else
        PADDLE_RANK_OPTION=""
    fi
    # 以下为通用执行命令，无特殊可不用修改
    case ${run_mode} in
    DP1-MP1-PP1) echo "run run_mode: DP1-MP1-PP1"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0 ${PADDLE_RANK_OPTION}\
            tools/auto.py -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp8.yaml \
            ${train_cmd}"
        workerlog_id=0
        ;;
    DP8-MP1-PP1) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            tools/auto.py -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp8.yaml \
            ${train_cmd}"
        workerlog_id=0
        ;;
    DP1-MP8-PP1|DP16-MP1-PP1) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            tools/auto.py -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_6.7B_sharding16.yaml -o Model.hidden_size=${hidden_size} \
            ${train_cmd}"
        workerlog_id=0
        ;;
    DP1-MP1-PP8) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            tools/auto.py -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_6.7B_sharding16.yaml \
            ${train_cmd}"
        workerlog_id=7
        ;;
    DP2-MP2-PP2) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            tools/auto.py -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_6.7B_sharding16.yaml \
            ${train_cmd}"
        workerlog_id_1=4
        workerlog_id_2=6
        ;;
    DP2-MP4-PP1) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            tools/auto.py -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_6.7B_sharding16.yaml \
            ${train_cmd}"
        workerlog_id_1=0
        workerlog_id_2=4
        ;;
    DP1-MP2-PP4|DP2-MP1-PP4) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            tools/auto.py -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_6.7B_sharding16.yaml \
            ${train_cmd}"
        workerlog_id_1=6
        workerlog_id_2=7
        ;;
    *) echo "choose run_mode "; exit 1;
    esac
    cd ../
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    if [[ ${model_item} =~ "CE" ]];then # CE精度-不限制执行时间
        ${train_cmd} > ${log_file} 2>&1
    else
        timeout 40m ${train_cmd} > ${log_file} 2>&1
    fi
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ ${device_num} != "N1C1" -a -d mylog ]; then
        case ${run_mode} in
        DP1-MP1-PP1|DP8-MP1-PP1|DP1-MP8-PP1|DP1-MP1-PP8|DP16-MP1-PP1) echo "${run_mode} cp  mylog/workerlog.${workerlog_id}"
            rm ${log_file}
            cp mylog/workerlog.${workerlog_id} ${log_file}
            ;;
        DP2-MP2-PP2|DP2-MP4-PP1|DP1-MP2-PP4|DP2-MP1-PP4) echo "${run_mode} cp  mylog/workerlog.${workerlog_id_1} & mylog/workerlog.${workerlog_id_2}"
            rm ${log_file}
            cp mylog/workerlog.${workerlog_id_1} ${log_file}
            cp mylog/workerlog.${workerlog_id_2} ${log_file}_2
            ;;
        *) echo "${run_mode} cp  mylog/workerlog.${workerlog_id}"
            rm ${log_file}
            workerlog_id=0
            cp mylog/workerlog.${workerlog_id} ${log_file}
            ;;
        esac
    fi
}

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
# 避免预分配的的显存影响实际值观测
export FLAGS_fraction_of_gpu_memory_to_use=0.1
# 旧执行器
export FLAGS_USE_STANDALONE_EXECUTOR=0
export FLAGS_CONVERT_GRAPH_TO_PROGRAM=0
export GLOG_v=0
# 新执行器
# export FLAGS_USE_STANDALONE_EXECUTOR=1
# export FLAGS_CONVERT_GRAPH_TO_PROGRAM=1
# export GLOG_v=0
# export FLAGS_new_executor_sequential_run=1

# bug begin_norm_axis
# sed -i '755,763d' /usr/local/lib/python3.7/dist-packages/paddle/distributed/auto_parallel/engine.py 
# sed -i  "/place = _get_device()/i\        auto_utils.initialize_pg_in_full_mode(all_process_groups, cur_rank)"  /usr/local/lib/python3.7/dist-packages/paddle/distributed/auto_parallel/engine.py 
source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开
