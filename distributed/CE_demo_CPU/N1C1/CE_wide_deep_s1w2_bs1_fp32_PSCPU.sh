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

model_item=CE_wide_deep_s1w2
dp_degree=1
mp_degree=1
pp_degree=1
bs_item=1
fp_item=fp32
run_mode=PSCPU
device_num=N1C1
server_num=1
worker_num=2

model=wide_deep
micro_bs=${bs_item}

bash ./distributed/CE_demo_CPU/benchmark_common/prepare.sh
# run
bash ./distributed/CE_demo_CPU/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${server_num} ${worker_num} 2>&1;
