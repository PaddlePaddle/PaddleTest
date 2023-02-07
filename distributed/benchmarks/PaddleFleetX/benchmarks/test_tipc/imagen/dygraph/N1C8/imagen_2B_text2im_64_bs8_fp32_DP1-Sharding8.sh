model_item=imagen_2B_text2im_64
dp_degree=1
mp_degree=1
pp_degree=1
bs_item=8
fp_item=fp32
run_mode=DP1-Sharding8
device_num=N1C8
yaml_path=ppfleetx/configs/multimodal/imagen/imagen_text2im_64x64_T5-11B.yaml
max_iter=1000
logging_freq=1
sharding_degree=8
sharding_stage=2

model=imagen
micro_bs=1

cd ./benchmarks
bash ./test_tipc/imagen/dygraph/benchmark_common/prepare.sh
# run
bash ./test_tipc/imagen/dygraph/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${yaml_path} ${max_iter} ${logging_freq} ${sharding_degree} ${sharding_stage} 2>&1;
