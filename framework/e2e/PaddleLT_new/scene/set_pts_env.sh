#!/usr/bin/env bash

# 最外层测试入口, 通过环境变量设定测试相关的一切信息(除了docker/whl/py以外)
export CASE_URL="${CASE_URL:-https://paddle-qa.bj.bcebos.com/PaddleLT/LayerCase/layercase.tar}"
export CASE_DIR=$(echo $(basename $CASE_URL) | cut -d '.' -f 1) # 设定子图case根目录
wget ${CASE_URL} --no-proxy && tar -xzf ${CASE_DIR}.tar
export TESTING="${TESTING:-yaml/dy^dy2stcinn_eval.yml}" # 设定测试项目配置yaml

echo "wheel_url is: ${wheel_url}"
echo "python_ver is: ${python_ver}"
echo "CASE_URL is: ${CASE_URL}"
echo "CASE_DIR is: ${CASE_DIR}"
echo "TESTING is: ${TESTING}"
