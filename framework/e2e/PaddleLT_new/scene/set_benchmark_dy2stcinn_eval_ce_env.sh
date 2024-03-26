#!/usr/bin/env bash

# 最外层测试入口, 通过环境变量设定测试相关的一切信息(除了docker/whl/py以外)
# export CASE_URL="${CASE_URL:-https://paddle-qa.bj.bcebos.com/PaddleLT/LayerCase/layercase.tar}"
# export CASE_DIR=$(echo $(basename $CASE_URL) | cut -d '.' -f 1) # 设定子图case根目录
# wget ${CASE_URL} --no-proxy && tar -xzf ${CASE_DIR}.tar

export CASE_DIR="${CASE_DIR:-perf_monitor}"
export TESTING="${TESTING:-yaml/dy2stcinn_eval_benchmark.yml}" # 设定测试项目配置yaml
export TESTING_MODE="${TESTING_MODE:-performance}" # 设定测试模型为性能
export PLT_SET_DEVICE="${PLT_SET_DEVICE:-gpu}"
export PLT_DEVICE_ID="${PLT_DEVICE_ID:-0}"

#性能测试专属环境变量
export PLT_BM_REPEAT="${PLT_BM_REPEAT:-10}"
export PLT_BM_STATIS="${PLT_BM_STATIS:-trimmean}"

#研发指定环境变量
export FLAGS_pir_apply_shape_optimization_pass=0
export FLAGS_enable_pir_api=1
export FLAGS_prim_enable_dynamic=true
export FLAGS_prim_all=true
export FLAGS_cinn_new_group_scheduler=1
export FLAGS_group_schedule_tiling_first=1
export FLAGS_cinn_bucket_compile=True
export FLAGS_cinn_compile_with_nvrtc=True
export FLAGS_nvrtc_compile_to_cubin=True
export FLAGS_support_reduce_stride_read=1

echo "wheel_url is: ${wheel_url}"
echo "python_ver is: ${python_ver}"
# echo "CASE_URL is: ${CASE_URL}"
echo "CASE_DIR is: ${CASE_DIR}"
echo "TESTING is: ${TESTING}"

echo "PLT_SET_DEVICE is: ${PLT_SET_DEVICE}"
echo "PLT_DEVICE_ID is: ${PLT_DEVICE_ID}"
