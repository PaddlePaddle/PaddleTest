#!/usr/bin/env bash
# 最外层执行脚本设定：环境变量、测试子图文件夹根目录、测试项目yaml

test_scene=$1
source ./set_env.sh $PTS_ENV_VARS  # 设定PTS环境变量
source ./set_docker_env.sh

docker_name="PaddleLayerTest_${AGILE_PIPELINE_BUILD_NUMBER}"

nvidia-docker run --rm -i --name ${docker_name} --privileged --shm-size=128g --net=host \
  -w /workspace \
  -v $PWD:/workspace \
  -e "AK=${AK}" -e "SK=${SK}" \
  -e "http_proxy=${http_proxy}" \
  -e "https_proxy=${https_proxy}" \
  -e "no_proxy=bcebos.com" \
  ${docker_image} /bin/bash -c "
ldconfig;

source ./set_env.sh $PTS_ENV_VARS  # 设定PTS环境变量
source ./${test_scene}
${python_ver} -m pip install -r requirement.txt
${python_ver} -m pip install ${wheel_url}
${python_ver} run.py
"
wait $!
exit $?
