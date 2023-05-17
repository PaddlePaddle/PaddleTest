# 两种使用场景，
# 一、是根据pipeline名称进行cuda版本以及python版本的判断，并根据字段判断是否需要打印相关warning，或者报错
# 二、根据输入的参数进行cuda版本以及python版本的判断
# 考虑预测库安装包地址的获取方式
# 
# 将两种编译产出的包分开处理，一种是CE编译任务产出的安装包，另一种是Nightly编译任务的安装包，区别是：Nightly安装包没有失败重试，但覆盖的python和cuda版本更全
## 或者是这样处理：
## 1. 如果CE任务中存在符合该版本的包，则把这个包作为推荐使用的安装包

# 退出码规定
# 1. 镜像相关退出码为10x
#   - 101 : 找不到指定版本的镜像名称
# 2. 安装包相关退出码为11x
#   - 115： 找不到符合cuda版本的安装包
    - 116

# 镜像信息

function DockerImages () {
    docker_type=$1
    cuda_version=$2
    DOCKER_EXIT_CODE=0

    if [[ ${docker_type} == "Centos" ]];then
        case ${cuda_version} in
        "Cuda102")
            echo "Selected Centos: Cuda102"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7.6-trt7.0-gcc8.2"
            ;;
        "Cuda112")
            echo "Selected Centos: Cuda112"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7.6-trt7.0-gcc8.2"
            ;;
        "Cuda116")
            echo "Selected Centos: Cuda116"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.6-cudnn8.4-trt8.4-gcc82"
            ;;
        "Cuda117")
            echo "Selected Centos: Cuda117"
            export Image_version="registry.baidubce.com/paddlepaddle/paddleqa:latest-dev-cuda11.7-cudnn8.4-trt8.4-gcc8.2-v1"
            ;;
        "Cuda118")
            echo "Selected Centos: Cuda118"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82"
            ;;
        *)
            DOCKER_EXIT_CODE=101
            ;;
        esac
    else
        docker_type="Ubuntu"
        case ${cuda_version} in
        "Cuda102")
            echo "Selected Ubuntu: Cuda102"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7.6-trt7.0-gcc8.2"
            ;;
        "Cuda112")
            echo "Selected Ubuntu: Cuda112"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.2-cudnn8.2-trt8.0-gcc82"
            ;;
        "Cuda116")
            echo "Selected Ubuntu: Cuda116"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.6-cudnn8.4.0-trt8.4.0.6-gcc82"
            ;;
        "Cuda117")
            echo "Selected Ubuntu: Cuda117"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.7-cudnn8.4-trt8.4-gcc8.2"
            ;;
        "Cuda118")
            echo "Selected Ubuntu: Cuda118"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.8-cudnn8.6-trt8.5-gcc8.2"
            ;;
        *)
            DOCKER_EXIT_CODE=101
            ;;
        esac
    fi
    if [[ "${DOCKER_EXIT_CODE}" == "101" ]];then
        echo "Could not find dockerimages that satisfy the requirement as follows:"
        echo "- Cuda Version: ${cuda_version}"
        echo "- Linux Type: ${docker_type}"
        echo "DOCKER_EXIT_CODE:${DOCKER_EXIT_CODE}"
    else
        echo "Image Name is ${Image_version}"
    fi
}

# 版本退场判断


# CPU安装包链接信息
function CpuPackageUrlInfo(){
    branch_info=$1
    package_version=$2

    case ${package_version} in
        "Python37")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-Cpu-LinuxCentos-Gcc82-MKL-Py37-Compile/latest/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl"
            ;;
        "Python38")
            if [[ "${branch_info}" == "Develop" ]];then
                export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-Cpu-LinuxCentos-Gcc82-OnInfer-Py38-Compile/latest/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl"
            else
                export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Cpu-Mkl-Avx-Gcc82/latest/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl"
            fi
            ;;
        "Python39")
                export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Cpu-Mkl-Avx-Gcc82/latest/paddlepaddle-0.0.0-cp39-cp39-linux_x86_64.whl"
            ;;
        "Python310")
                export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Cpu-Mkl-Avx-Gcc82/latest/paddlepaddle-0.0.0-cp310-cp310-linux_x86_64.whl"
            ;;
        "Inference")
            # 预测库安装包的地址
            if [[ "${branch_info}" == "Develop" ]];then
                export paddle_inference="https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-Cpu-LinuxCentos-Gcc82-OnInfer-Py38-Compile/latest/paddle_inference.tgz"
                export paddle_inference_c="https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-Cpu-LinuxCentos-Gcc82-OnInfer-Py38-Compile/latest/paddle_inference_c.tgz"
            else
                export WHELLINFO_EXITCODE=109
            fi
            ;;
        *)
            export WHELLINFO_EXITCODE=107
            ;;
    esac
}

# CUDA10.2安装包链接信息
function Cu102PackageUrlInfo(){
    branch_info=$1
    package_version=$2

    case ${package_version} in
        "Python37")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-GpuAll-LinuxCentos-Gcc82-Cuda102-Trtoff-Py37-Compile/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"
            ;;
        "Python38")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-GpuAll-Centos-Gcc82-Cuda102-Cudnn76-Trt6018-Py38-Compile/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl"
            ;;
        "Python39")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda10.2-Cudnn7-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post102.post102-cp39-cp39-linux_x86_64.whl"
            ;;
        "Python310")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda10.2-Cudnn7-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post102.post102-cp310-cp310-linux_x86_64.whl"
            ;;
        "Inference")
            # 预测库安装包的地址
            export paddle_inference="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-GpuAll-Centos-Gcc82-Cuda102-Cudnn76-Trt6018-Py38-Compile/latest/paddle_inference.tgz"
            export paddle_inference_c="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-GpuAll-Centos-Gcc82-Cuda102-Cudnn76-Trt6018-Py38-Compile/latest/paddle_inference_c.tgz"
            ;;
        *)
            export WHELLINFO_EXITCODE=117
            ;;
    esac
}

# Cuda112安装包链接信息
function Cu112PackageUrlInfo(){
    branch_info=$1
    package_version=$2

    case ${package_version} in
        "Python37")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.2-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post112-cp37-cp37m-linux_x86_64.whl"
            ;;
        "Python38")
            echo "Recomend!!!"
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-GpuAll-LinuxCentos-Gcc82-Cuda112-Trtoff-Py38-Compile/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl"
            ;;
        "Python39")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.2-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post112-cp39-cp39-linux_x86_64.whl"
            ;;
        "Python310")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.2-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post112-cp310-cp310-linux_x86_64.whl"
            ;;
        "Inference")
            # 预测库安装包的地址
            export paddle_inference="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-GpuAll-Centos-Gcc82-Cuda112-Cudnn82-Trt8034-Py38-Compile/latest/paddle_inference.tgz"
            export paddle_inference_c="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-GpuAll-Centos-Gcc82-Cuda112-Cudnn82-Trt8034-Py38-Compile/latest/paddle_inference_c.tgz"
            ;;
        *)
            export WHELLINFO_EXITCODE=117
            ;;
    esac
}

# Cuda116安装包链接信息
function Cu116PackageUrlInfo(){
    branch_info=$1
    package_version=$2

    case ${package_version} in
        "Python37")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.6-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post116-cp37-cp37m-linux_x86_64.whl"
            ;;
        "Python38")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.6-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post116-cp38-cp38-linux_x86_64.whl"
            ;;
        "Python39")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.6-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-linux_x86_64.whl"
            ;;
        "Python310")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.6-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post116-cp310-cp310-linux_x86_64.whl"
            ;;
        *)
            export WHELLINFO_EXITCODE=117
            ;;
    esac
}

# Cuda117安装包链接信息
function Cu117PackageUrlInfo(){
    branch_info=$1
    package_version=$2

    case ${package_version} in
        "Python37")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.7-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post117-cp37-cp37m-linux_x86_64.whl"
            ;;
        "Python38")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.7-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post117-cp38-cp38-linux_x86_64.whl"
            ;;
        "Python39")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-GpuAll-LinuxCentos-Gcc82-Cuda117-Cudnn84-Trt84-Py39-Compile/latest/paddlepaddle_gpu-0.0.0-cp39-cp39-linux_x86_64.whl"
            ;;
        "Python310")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.7-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-linux_x86_64.whl"
            ;;
        *)
            export WHELLINFO_EXITCODE=117
            ;;
    esac
}

# Cuda118安装包链接信息
function Cu118PackageUrlInfo(){
    branch_info=$1
    package_version=$2

    case ${package_version} in
        "Python37")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.8-Cudnn8.6-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post118-cp37-cp37m-linux_x86_64.whl"
            ;;
        "Python38")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.8-Cudnn8.6-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post118-cp38-cp38-linux_x86_64.whl"
            ;;
        "Python39")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.8-Cudnn8.6-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post118-cp39-cp39-linux_x86_64.whl"
            ;;
        "Python310")
            export paddle_whl="https://paddle-qa.bj.bcebos.com/paddle-pipeline/${branch_info}-TagBuild-Training-Linux-Gpu-Cuda11.8-Cudnn8.6-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post118-cp310-cp310-linux_x86_64.whl"
            ;;
        *)
            export WHELLINFO_EXITCODE=117
            ;;
    esac
}


# 安装包链接
function WheelUrlInfo(){
    branch_info=${1:-"Develop"}
    cuda_version=$2
    python_version=$3
    infer_package=${4:-"N"}
    # branch信息默认为Develop分支
    branch_info=${3:-"Develop"}

    WHELLINFO_EXITCODE=0

    if [[ "${branch_info}" == "Release" ]];then
        echo "The branch information you selected is ${branch_info};"
    elif [[ "${branch_info}" == "Develop" ]]
        echo "The branch information you selected is ${branch_info};"
    else
        echo "The branch information input is wrong, you can choose to enter 'Develop' or 'Release';"
        WHELLINFO_EXITCODE=102
        exit ${WHELLINFO_EXITCODE}
    fi

    case ${cuda_version} in
        "Cpu")
            CpuPackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "N" ]];then
                CpuPackageUrlInfo ${branch_info} Inference
            fi
            ;;
        "Cuda102")
            Cu102PackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "N" ]];then
                Cu102PackageUrlInfo ${branch_info} Inference
            fi
            ;;
        "Cuda112")
            Cu112PackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "N" ]];then
                Cu112PackageUrlInfo ${branch_info} Inference
            fi
            ;;
        "Cuda116")
            Cu116PackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "N" ]];then
                Cu116PackageUrlInfo ${branch_info} Inference
            fi
            ;;
        "Cuda117")
            Cu117PackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "N" ]];then
                Cu117PackageUrlInfo ${branch_info} Inference
            fi
            ;;
        "Cuda118")
            Cu118PackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "N" ]];then
                Cu118PackageUrlInfo ${branch_info} Inference
            fi
            ;;
        *)
            DOCKER_EXIT_CODE=115
            ;;
        esac

    if [[ "${WHELLINFO_EXITCODE}" == "107" ]];then
        echo "Could not find packages that satisfy the requirement as follows:"
        echo "- Cuda Version: ${cuda_version}"
        echo "- Python Version: ${package_version}"
        echo "- Branch Info:${branch_info}"
    else
        echo " ${Image_version}"
    fi
}