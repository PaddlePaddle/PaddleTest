#!/usr/bin/env bash

export CASE_URL="${CASE_URL:-https://paddle-qa.bj.bcebos.com/PaddleLT/LayerCase/layercase.tar}"
export CASE_DIR=$(echo $(basename $CASE_URL) | cut -d '.' -f 1) # 设定子图case根目录
wget ${CASE_URL} --no-proxy && tar -xzf ${CASE_DIR}.tar
export TESTING="${TESTING:-yaml/dy^dy2stcinn_eval.yml}" # 设定测试项目配置yaml
export wheel_url="${wheel_url:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-LinuxCentos-Gcc82-Cuda112-Trton-Py38-Compile/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl}"
export python_ver="${python_ver:-python3.8}"

echo "wheel_url is: ${wheel_url}"
echo "python_ver is: ${python_ver}"
echo "CASE_URL is: ${CASE_URL}"
echo "CASE_DIR is: ${CASE_DIR}"
echo "TESTING is: ${TESTING}"
