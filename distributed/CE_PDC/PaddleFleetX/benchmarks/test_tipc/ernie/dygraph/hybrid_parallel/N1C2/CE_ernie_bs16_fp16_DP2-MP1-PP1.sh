model_item=CE_ernie
dp_degree=2
mp_degree=1
pp_degree=1
bs_item=16
fp_item=fp16
run_mode=DP2-MP1-PP1
device_num=N1C2
max_iter=50000

model=ernie
micro_bs=8

cd ./benchmarks
bash ./test_tipc/ernie/dygraph/hybrid_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/ernie/dygraph/hybrid_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${max_iter} 2>&1;
