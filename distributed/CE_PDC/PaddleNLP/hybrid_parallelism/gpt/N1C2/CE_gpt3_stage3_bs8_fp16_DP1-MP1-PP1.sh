model_item=CE_gpt3_stage3
dp_degree=1
mp_degree=1
pp_degree=1
bs_item=8
fp_item=fp16
run_mode=DP1-MP1-PP1
device_num=N1C2
max_iter=20000
use_sharding=true
use_recompute=True
sharding_stage=3
sharding_offload=True
eval_freq=20000
sharding_degree=2  

model=gpt
micro_bs=4

cd ./tests
bash ./test_tipc/dygraph/hybrid_parallelism/${model}/benchmark_common/prepare.sh
# run
bash ./test_tipc/dygraph/hybrid_parallelism/${model}/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${max_iter} ${use_sharding} ${use_recompute} ${sharding_stage} ${sharding_offload} ${eval_freq} ${sharding_degree} 2>&1;
