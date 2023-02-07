model_item=CE_gpt_stage2
dp_degree=1
mp_degree=1
pp_degree=1
bs_item=16
fp_item=fp16
run_mode=DP1-MP1-PP1-Sharding2
device_num=N1C2
sharding_degree=2
sharding_stage=2
sharding_offload=True
max_iter=20000
eval_freq=20000

model=gpt
micro_bs=8

cd ./benchmarks
bash ./test_tipc/gpt/dygraph/sharding/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/dygraph/sharding/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${sharding_degree} ${sharding_stage} ${sharding_offload} ${max_iter} ${eval_freq} 2>&1;
