#! /bin/bash

set -ex

DOCKER_NAME="test_infer_slim"
# DOCKER_IMAGE="paddlepaddle/paddle_manylinux_devel:cuda11.1-cudnn8.1-gcc82-trt7"
# PADDLE_WHL="https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release_GpuAll_LinuxCentos_Gcc82_Cuda11.1_cudnn8.1.1_trt8406_Py38_Compile_H/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl"
DOCKER_IMAGE=${DOCKER_IMAGE:-registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.2-cudnn8.1-trt8.0-gcc8.2}
PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-Centos-Gcc82-Cuda112-Cudnn82-Trt8034-Py38-Compile/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl}
PADDLE_BRANCH=${PADDLE_BRANCH:-release/2.4}
DEVICE=${DEVICE:-T4}
MODE=${MODE:-trt_int8,trt_fp16,mkldnn_int8,mkldnn_fp32}
METRIC=${METRIC:-jingdu,xingneng}

export CUDA_SO="$(\ls -d /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls -d /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls -d /dev/nvidia* | xargs -I{} echo '--device {}:{}')

docker rm -f ${DOCKER_NAME} || echo "remove docker ""${DOCKER_NAME}"" failed"
nvidia-docker run -i --rm \
    --name ${DOCKER_NAME} \
    --privileged \
    --net=host \
    --shm-size=128G \
    -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi ${CUDA_SO} ${DEVICES} \
    -v $(pwd):/workspace \
    -w /workspace \
    -e "LANG=en_US.UTF-8" \
    -e "PYTHONIOENCODING=utf-8" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e "no_proxy=bcebos.com,goproxy.cn,baidu.com,bcebos.com" \
    -e PADDLE_WHL=${PADDLE_WHL} \
    -e PADDLE_BRANCH=${PADDLE_BRANCH} \
    -e DOCKER_IMAGE=${DOCKER_IMAGE} \
    -e DEVICE=${DEVICE} \
    -e MODE=${MODE} \
    --net=host \
    ${DOCKER_IMAGE} \
     /bin/bash -c -x '

export MODE=${MODE}
export METRIC=${METRIC}

export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.8.0/bin/python3.8 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.8.0/include/python3.8 -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.8.0/lib/libpython3.so"

wget https://paddle-qa.bj.bcebos.com/tools/TensorRT-8.4.0.6.tgz
tar -zxf TensorRT-8.4.0.6.tgz
export LD_LIBRARY_PATH=${PWD}/TensorRT-8.4.0.6/lib/:${LD_LIBRARY_PATH}


python -m pip install --retries 50 --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip config set global.index-url https://mirror.baidu.com/pypi/simple;

pip install -r requirements.txt

pip install -U ${PADDLE_WHL}

pip install nvidia-pyindex
pip install nvidia-cublas-cu11
pip install nvidia-tensorrt
pip install pycuda
pip install openpyxl

bash run.sh

PADDLE_COMMIT=`python -c 'import paddle; print(paddle.version.commit)'`
DT=`date "+%Y-%m-%d"`
SAVE_FILE=${DT}_${PADDLE_BRANCH}_${PADDLE_COMMIT}.xlsx

python get_benchmark_info.py ${DOCKER_IMAGE} ${PADDLE_BRANCH} ${PADDLE_COMMIT} ${DEVICE} ${MODE} ${METRIC} ${SAVE_FILE}

UPLOAD_FILE_PATH=`pwd`/${SAVE_FILE}

# pipeline 工具需提前下载到该路径中
cd pipeline
python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple --trusted-host mirror.baidu.com;
python upload.py --bucket_name paddle-qa --object_key inference_benchmark/paddle/slim/${SAVE_FILE} --upload_file_name ${UPLOAD_FILE_PATH}

cat "https://paddle-qa.bj.bcebos.com/inference_benchmark/paddle/slim/${SAVE_FILE}" >benchmark_res_url.txt

'
