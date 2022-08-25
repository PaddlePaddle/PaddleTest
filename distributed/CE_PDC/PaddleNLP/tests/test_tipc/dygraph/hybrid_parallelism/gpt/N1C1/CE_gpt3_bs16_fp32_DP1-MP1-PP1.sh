model_item=CE_gpt3
dp_degree=1
mp_degree=1
pp_degree=1
bs_item=16
fp_item=fp32
run_mode=DP1-MP1-PP1
device_num=N1C1
max_iter=50000

model=gpt
micro_bs=2

cd ./tests
bash ./test_tipc/dygraph/hybrid_parallelism/${model}/benchmark_common/prepare.sh
# run
bash ./test_tipc/dygraph/hybrid_parallelism/${model}/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${max_iter} 2>&1;
