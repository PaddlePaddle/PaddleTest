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

# 以下为run_benchmark.sh的必选参数
param="model_item=facebook-llama-13b_seqlen2048_pretrain "
param+="global_batch_size=32 "
param+="fp_item=bf16 "
param+="run_mode=MP2-PP4-vpp5-mbs2-acc16-recompute "
param+="device_num=N1C8 "
# 以下为run_benchmark.sh的可选参数
param+="sequence_parallel=0 "
param+="pp_recompute_interval=1 "
param+="tensor_parallel_config=enable_mp_async_allreduce,enable_mp_skip_c_identity,enable_mp_fused_linear_param_grad_add "
param+="recompute_use_reentrant=true "

cd ./tests
bash ./test_tipc/dygraph/hybrid_parallelism/llama/benchmark_common/prepare.sh

bash -c "${param} bash ./test_tipc/dygraph/hybrid_parallelism/llama/benchmark_common/run_benchmark.sh"
