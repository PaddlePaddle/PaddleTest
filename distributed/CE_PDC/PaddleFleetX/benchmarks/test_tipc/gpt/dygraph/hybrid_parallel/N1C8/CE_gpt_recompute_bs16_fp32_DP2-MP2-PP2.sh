model_item=CE_gpt_recompute
dp_degree=2
mp_degree=2
pp_degree=2
bs_item=16
fp_item=fp32
run_mode=DP2-MP2-PP2
device_num=N1C8
max_iter=50000
use_sharding=false
use_recompute=True

model=gpt
micro_bs=2

cd ./benchmarks
bash ./test_tipc/gpt/dygraph/hybrid_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/dygraph/hybrid_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${max_iter} ${use_sharding} ${use_recompute} 2>&1;
