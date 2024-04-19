#!/usr/bin/env bash

# 最外层测试入口, 通过环境变量设定测试相关的一切信息(除了docker/whl/py以外)
export CASE_TYPE="${CASE_TYPE:-layerApicase}"
export CASE_DIR="${CASE_DIR:-perf_monitor}"
export TESTING="${TESTING:-yaml/dy2stcinn_eval_benchmark.yml}" # 设定测试项目配置yaml
export TESTING_MODE="${TESTING_MODE:-performance}" # 设定测试模式为性能
export PLT_SET_DEVICE="${PLT_SET_DEVICE:-gpu}" # 硬件
export PLT_DEVICE_ID="${PLT_DEVICE_ID:-6}"  # 设备编号
export CUDA_VISIBLE_DEVICES="${PLT_DEVICE_ID:-6}"
export FRAMEWORK="${FRAMEWORK:-PaddlePaddle}"  # 框架种类
export MULTI_WORKER="${MULTI_WORKER:-0}"  # 并行数

#性能测试专属环境变量
export PLT_BM_MODE="${PLT_BM_MODE:-latest}"  #基线任务为baseline, 测试任务为latest
export PLT_BM_DB="${PLT_BM_DB:-select}"  # insert: 存入数据, 作为基线或对比; select: 不存数据, 仅对比并生成表格; non-db: 不加载数据库，仅生成表格
export PLT_BM_EMAIL="${PLT_BM_EMAIL:-False}"  # True: 发送邮件  False: 不发送邮件
export PLT_BM_REPEAT="${PLT_BM_REPEAT:-1000}"  # 性能测试重复轮次
export TIMEIT_NUM="${TIMEIT_NUM:-1}"  # timeit number数
export PLT_BM_STATIS="${PLT_BM_STATIS:-trimmean}"  # 统计策略trimmean, mean, best, best_top_k

#研发指定环境变量
export FLAGS_pir_apply_shape_optimization_pass=0
export FLAGS_enable_pir_api=1
export FLAGS_cinn_new_group_scheduler=1
export FLAGS_group_schedule_tiling_first=1
export FLAGS_cinn_bucket_compile=True
export FLAGS_cinn_compile_with_nvrtc=True
export FLAGS_nvrtc_compile_to_cubin=True
export FLAGS_support_reduce_stride_read=1

echo "wheel_url is: ${wheel_url}"
echo "python_ver is: ${python_ver}"
echo "CASE_DIR is: ${CASE_DIR}"
echo "TESTING is: ${TESTING}"
echo "CUDA_VISIBLE_DEVICES is: ${CUDA_VISIBLE_DEVICES}"

echo "TESTING_MODE is: ${TESTING_MODE}"
echo "PLT_SET_DEVICE is: ${PLT_SET_DEVICE}"
echo "PLT_DEVICE_ID is: ${PLT_DEVICE_ID}"
echo "FRAMEWORK is: ${FRAMEWORK}"
echo "MULTI_WORKER is: ${MULTI_WORKER}"
echo "PLT_BM_MODE is: ${PLT_BM_MODE}"
echo "PLT_BM_DB is: ${PLT_BM_DB}"
echo "PLT_BM_REPEAT is: ${PLT_BM_REPEAT}"
echo "PLT_BM_STATIS is: ${PLT_BM_STATIS}"
