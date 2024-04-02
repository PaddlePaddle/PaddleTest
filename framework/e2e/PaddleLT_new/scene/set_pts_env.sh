#!/usr/bin/env bash

# 最外层测试入口, 通过环境变量设定测试相关的一切信息(除了docker/whl/py以外)
export CASE_DIR="${CASE_DIR:-sublayer160}"
export TESTING="${TESTING:-yaml/dy^dy2stcinn_eval.yml}" # 设定测试项目配置yaml
export TESTING_MODE="${TESTING_MODE:-precision}" # 设定测试模型为精度
export PLT_SET_DEVICE="${PLT_SET_DEVICE:-gpu}"
export PLT_DEVICE_ID="${PLT_DEVICE_ID:-0}"
export FRAMEWORK="${FRAMEWORK:-PaddlePaddle}"
export THREAD_WORKER="${THREAD_WORKER:-0}"

echo "wheel_url is: ${wheel_url}"
echo "python_ver is: ${python_ver}"
echo "CASE_DIR is: ${CASE_DIR}"
echo "TESTING is: ${TESTING}"

echo "TESTING_MODE is: ${TESTING_MODE}"
echo "PLT_SET_DEVICE is: ${PLT_SET_DEVICE}"
echo "PLT_DEVICE_ID is: ${PLT_DEVICE_ID}"
echo "FRAMEWORK is: ${FRAMEWORK}"
echo "THREAD_WORKER is: ${THREAD_WORKER}"
