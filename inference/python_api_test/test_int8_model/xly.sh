#! /bin/bash

set -ex

DOCKER_NAME="test_infer_slim"
DOCKER_IMAGE="paddlepaddle/paddle_manylinux_devel:cuda11.1-cudnn8.1-gcc82-trt7"
PADDLE_WHL="https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release_GpuAll_LinuxCentos_Gcc82_Cuda11.1_cudnn8.1.1_trt8406_Py38_Compile_H/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl"

# define version compare function
function version_lt() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"; }

# check nvidia-docker version
nv_docker_version=`nvidia-docker version | grep NVIDIA | cut -d" " -f3`
if version_lt ${nv_docker_version} 2.0.0; then
    echo -e "nv docker version ${nv_docker_version} is less than 2.0.0, should map CUDA_SO and DEVICES to docker"
    export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    export DEVICES=$(\ls -d /dev/nvidia* | xargs -I{} echo '--device {}:{}')
fi

docker rm -f ${DOCKER_NAME} || echo "remove docker ""${DOCKER_NAME}"" failed"
nvidia-docker run -i --rm \
    --name ${DOCKER_NAME} \
    --privileged \
    --net=host \
    --shm-size=128G \
    -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi ${CUDA_SO} ${DEVICES} \
    -v /home/disk1/:/home/disk1/ \
    -v $(pwd):/workspace \
    -w /workspace \
    -e "LANG=en_US.UTF-8" \
    -e "PYTHONIOENCODING=utf-8" \
    -e "no_proxy=bcebos.com,goproxy.cn,baidu.com,bcebos.com" \
    --net=host \
    ${DOCKER_IMAGE} \
     /bin/bash -c -x "


export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
export PYTHON_FLAGS='-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.8.0/bin/python3.8 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.8.0/include/python3.8 -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.8.0/lib/libpython3.so'

export LD_LIBRARY_PATH=${PWD}/TensorRT-8.4.0.6/lib/:${LD_LIBRARY_PATH}


python -m pip install --retries 50 --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip config set global.index-url https://mirror.baidu.com/pypi/simple;

pip install -r requirements.txt

pip install -U ${PADDLE_WHL}

bash run.sh
"
