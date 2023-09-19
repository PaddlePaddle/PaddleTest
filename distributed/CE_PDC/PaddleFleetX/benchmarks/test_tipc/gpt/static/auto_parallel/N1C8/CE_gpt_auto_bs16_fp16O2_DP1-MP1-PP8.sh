model_item=CE_gpt_auto
dp_degree=1
mp_degree=1
pp_degree=8
bs_item=16
fp_item=fp16O2
run_mode=DP1-MP1-PP8
device_num=N1C8
num_workers=7
max_iter=50000
use_recompute=False

model=gpt
micro_bs=2

cd ./benchmarks
bash ./test_tipc/gpt/static/auto_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/static/auto_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${num_workers} ${max_iter} ${use_recompute} 2>&1;
