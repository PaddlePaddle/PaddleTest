#!/usr/bin/env bash

# a100性能最外层执行脚本
test_scene=$1
source ./PTSTools/tools/set_env/set_env.sh ${PTS_ENV_VARS}  # 设定PTS环境变量
source ./set_docker_env.sh # 设定docker环境相关参数

docker_name="PaddleLayerTest_perf_${AGILE_PIPELINE_BUILD_NUMBER}"
export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES="$(\ls -d /dev/nvidia* | xargs -I{} echo '-v {}:{}') $(\ls /dev/nvidia-caps/* | xargs -I{} echo '-v {}:{}')"
RUN_IMAGE_NAME=iregistry.baidu-int.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.8-cudnn8.6-nccl2.15.5

docker run --rm -i --name=${docker_name} --network=host --privileged -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
${DEVICES} ${CUDA_SO} -v ${PWD}:/workspace \
  -w /workspace \
  -e "AK=${AK}" -e "SK=${SK}" \
  -e "http_proxy=${http_proxy}" \
  -e "https_proxy=${https_proxy}" \
  -e "no_proxy=bcebos.com" \
  -e "PLT_DEVICE_ID=${PLT_DEVICE_ID}" \
  -e "python_ver=${python_ver}" \
  -e "wheel_url=${wheel_url}" \
  -e "AGILE_PIPELINE_BUILD_ID=${AGILE_PIPELINE_BUILD_ID}" \
  --shm-size=128G \
  ${RUN_IMAGE_NAME} \
  /bin/bash -c "
ldconfig;

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
source ./PTSTools/tools/set_env/set_env.sh ${PTS_ENV_VARS}  # 设定PTS环境变量
source ./${test_scene}
${python_ver} -m pip install -r requirement.txt
${python_ver} -m pip install -r ./PTSTools/LogParseUpload/requirement.txt;
cp -r ./PTSTools/Uploader/apibm_config.yml .
${python_ver} -m pip install ${wheel_url}

unset http_proxy && unset https_proxy;
cp -r ./PTSTools/LogParseUpload/callback.sh .
${python_ver} run.py
"
