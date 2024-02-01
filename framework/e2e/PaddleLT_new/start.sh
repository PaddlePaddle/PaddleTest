#!/usr/bin/env bash

# 最外层执行脚本
test_scene=$1
source ./PTSTools/tools/set_env/set_env.sh ${PTS_ENV_VARS}  # 设定PTS环境变量
source ./set_docker_env.sh # 设定docker环境相关参数

docker_name="PaddleLayerTest_${AGILE_PIPELINE_BUILD_NUMBER}"

nvidia-docker run --rm -i --name ${docker_name} --privileged --shm-size=128g --net=host \
  -w /workspace \
  -v $PWD:/workspace \
  -e "AK=${AK}" -e "SK=${SK}" \
  -e "http_proxy=${http_proxy}" \
  -e "https_proxy=${https_proxy}" \
  -e "no_proxy=bcebos.com" \
  -e "python_ver=${python_ver}" \
  -e "wheel_url=${wheel_url}" \
  ${docker_image} /bin/bash -c "
ldconfig;

source ./PTSTools/tools/set_env/set_env.sh ${PTS_ENV_VARS}  # 设定PTS环境变量
source ./${test_scene}
${python_ver} -m pip install -r requirement.txt
${python_ver} -m pip install -r ./PTSTools/LogParseUpload/requirement.txt;
${python_ver} -m pip install ${wheel_url}
${python_ver} run.py

cp -r report ./PTSTools/LogParseUpload;
cd ./PTSTools/LogParseUpload;
unset http_proxy && unset https_proxy;

${python_ver} upload.py --file_path report --id ${pts_id} --status '成功'
"
