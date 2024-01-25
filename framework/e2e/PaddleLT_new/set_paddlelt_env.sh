#!/usr/bin/env bash

export CASE_URL="${CASE_URL:-https://paddle-qa.bj.bcebos.com/PaddleLT/LayerCase/demo_case.tar}"
export CASE_DIR=$(echo $(basename $CASE_URL) | cut -d '.' -f 1) # 设定子图case根目录
wget -q ${CASE_URL} --no-proxy && tar -xzf ${CASE_DIR}.tar
export TESTING="${TESTING:-yaml/dy^dy2stcinn_eval.yml}" # 设定测试项目配置yaml
export docker_type="${docker_type:-Ubuntu}"
export cuda_ver="${cuda_ver:-cuda11.2}"
export wheel_url="${wheel_url:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-LinuxCentos-Gcc82-Cuda110-Trtoff-Py38-Compile/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl}"
export python_ver="${python_ver:-python3.8}"

if [[ "${docker_type}" == "Ubuntu" ]];then
    case ${cuda_ver} in
    "cuda10.2")
        echo "Selected Ubuntu: Cuda10.2"
        export docker_image="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2"
        ;;
    "cuda11.2")
        echo "Selected Ubuntu: Cuda11.2"
        export docker_image="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8.2-trt8.0-gcc82"
        ;;
    "cuda11.6")
        echo "Selected Ubuntu: Cuda11.6"
        export docker_image="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.6-cudnn8.4-trt8.4-gcc82"
        ;;
    "cuda11.7")
        echo "Selected Ubuntu: Cuda11.7"
        export docker_image="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.7-cudnn8.4-trt8.4-gcc82"
        ;;
    "cuda11.8")
        echo "Selected Ubuntu: Cuda11.8"
        export docker_image="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82"
        ;;
    "cuda12.0")
        echo "Selected Ubuntu: Cuda12.0"
        export docker_image="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda12.0-cudnn8.9-trt8.6-gcc12.2"
        ;;
    *)
        echo "Unknown CUDA version: ${cuda_ver}"
        exit 1
        ;;
    esac

elif [[ "${docker_type}" == "Centos" ]];then
    case ${cuda_ver} in
    "cuda10.2")
        echo "Selected Centos: Cuda102"
        export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7.6-trt7.0-gcc8.2"
        ;;
    "cuda11.2")
        echo "Selected Centos: Cuda112"
        export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.2-cudnn8.2-trt8.0-gcc82"
        ;;
    "cuda11.6")
        echo "Selected Centos: Cuda116"
        export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.6-cudnn8.4-trt8.4-gcc8.2"
        ;;
    "cuda11.7")
        echo "Selected Centos: Cuda117"
        export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.7-cudnn8.4-trt8.4-gcc8.2"
        ;;
    "cuda11.8")
        echo "Selected Centos: Cuda118"
        export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.8-cudnn8.6-trt8.5-gcc8.2"
        ;;
    "cuda12.0")
        echo "Selected Centos: Cuda120"
        export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda12.0-cudnn8.9-trt8.6-gcc12.2"
        ;;
    *)
        echo "Unknown CUDA version: ${cuda_ver}"
        exit 1
        ;;
    esac
else
    echo "Unknown docker_type: ${docker_type}"
    exit 1
fi

echo "wheel_url is: ${wheel_url}"
echo "python_ver is: ${python_ver}"
echo "CASE_DIR is: ${CASE_DIR}"
echo "TESTING is: ${TESTING}"
