model_item=CE_gpt3_moe
dp_degree=1
sharding_degree=8
bs_item=8
fp_item=fp32
run_mode=Sharding_MoE_C8
device_num=N1C8
num_expert=8
max_iter=10000

model=gpt

# get data
cd ./tests
bash ./test_tipc/dygraph/moe/${model}/benchmark_common/prepare.sh
# run
bash ./test_tipc/dygraph/moe/${model}/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${sharding_degree} ${bs_item} ${run_mode} ${device_num} ${num_expert} ${max_iter} 2>&1;
