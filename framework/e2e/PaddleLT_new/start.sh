#!/usr/bin/env bash
# 最外层执行脚本，设定：环境变量、测试子图文件夹根目录、测试项目yaml

source set_env.sh $PTS_ENV_VARS  # 设定PTS环境变量
source set_paddlelt_env.sh # 设定PaddleLT环境变量(docker image, python, wheel_url等默认值)


docker_name="PaddleLayerTest_${AGILE_PIPELINE_BUILD_NUMBER}"

nvidia-docker run --rm -i --name ${docker_name} --privileged --shm-size=128g --net=host \
  -w /workspace \
  -v $PWD:/workspace \
  -e "AK=${AK}" -e "SK=${SK}" \
  -e "http_proxy=${http_proxy}" \
  -e "https_proxy=${https_proxy}" \
  -e "no_proxy=bcebos.com" \
  ${docker_image} /bin/bash -c '
ldconfig;

${python_ver} run.py
'
wait $!
exit $?
