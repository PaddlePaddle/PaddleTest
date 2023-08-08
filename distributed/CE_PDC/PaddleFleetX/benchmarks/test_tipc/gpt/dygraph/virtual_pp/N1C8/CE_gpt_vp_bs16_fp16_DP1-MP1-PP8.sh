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

model_item=CE_gpt_vp
dp_degree=1
mp_degree=1
pp_degree=8
bs_item=16
fp_item=fp16
run_mode=DP1-MP1-PP8
device_num=N1C8
max_iter=100
virtual_pp_degree=2

model=gpt
micro_bs=1

cd ./benchmarks
bash ./test_tipc/gpt/dygraph/virtual_pp/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/dygraph/virtual_pp/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${max_iter} ${virtual_pp_degree} 2>&1;
