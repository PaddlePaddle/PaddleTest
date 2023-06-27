# 两种使用场景，
# 一、是根据pipeline名称进行cuda版本以及python版本的判断，并根据字段判断是否需要打印相关warning，或者报错
# 二、根据输入的参数进行cuda版本以及python版本/预测库的判断
#
# 将两种编译产出的包分开处理，一种是CE编译任务产出的安装包，另一种是Nightly编译任务的安装包，区别是：Nightly安装包没有失败重试，但覆盖的python和cuda版本更全
## 或者是这样处理：
## 1. 如果CE任务中存在符合该版本的包，则把这个包作为推荐使用的安装包

# 退出码规定
# 1. 镜像相关退出码为10x
#   - 101 : 找不到指定版本的镜像名称
# 2. 安装包相关退出码为11x
#   - 114: 分支信息有误
#   - 115：找不到符合cuda版本的安装包
#   - 116: 找不到某个python版本或者inference的包

set +x
set -e

# 镜像信息
function DockerImages () {
    docker_type=$1
    cuda_version=$2
    DOCKER_EXIT_CODE=0
    # 增加Manual模式，如果已经手动设置了安装包链接，不执行后续逻辑
    if [[ "${Image_version}" != "" ]];then
        docker_type="Manual"
    fi

    if [[ "${docker_type}" == "Manual" ]];then
        echo "Select Manual Mode!!!"
    elif [[ "${docker_type}" == "Centos" ]];then
        case ${cuda_version} in
        "Cuda102")
            echo "Selected Centos: Cuda102"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7.6-trt7.0-gcc8.2"
            ;;
        "Cuda112")
            echo "Selected Centos: Cuda112"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.2-cudnn8.2-trt8.0-gcc82"
            ;;
        "Cuda116")
            echo "Selected Centos: Cuda116"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.6-cudnn8.4-trt8.4-gcc8.2"
            ;;
        "Cuda117")
            echo "Selected Centos: Cuda117"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.7-cudnn8.4-trt8.4-gcc8.2"
            ;;
        "Cuda118")
            echo "Selected Centos: Cuda118"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.8-cudnn8.6-trt8.5-gcc8.2"
            ;;
        "Cuda120")
            echo "Selected Centos: Cuda120"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda12.0-cudnn8.9-trt8.6-gcc12.2"
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
            export Image_version="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2"
            ;;
        "Cuda112")
            echo "Selected Ubuntu: Cuda112"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8.2-trt8.0-gcc82"
            ;;
        "Cuda116")
            echo "Selected Ubuntu: Cuda116"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.6-cudnn8.4-trt8.4-gcc82"
            ;;
        "Cuda117")
            echo "Selected Ubuntu: Cuda117"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.7-cudnn8.4-trt8.4-gcc82"
            ;;
        "Cuda118")
            echo "Selected Ubuntu: Cuda118"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82"
            ;;
        "Cuda120")
            echo "Selected Ubuntu: Cuda120"
            export Image_version="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda12.0-cudnn8.9-trt8.6-gcc12.2"
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
        exit ${DOCKER_EXIT_CODE}
    else
        echo "Image Name is ${Image_version}"
        echo 'Ps. You can get this image through "${Image_version}";'
    fi
}

# 版本退场判断
function VersionExitJudgment () {
    cuda_version=$1
    python_version=$2
    branch_info=$3
    if [[ "${branch_info}" == "Develop" ]];then
        case ${cuda_version} in
            "Cpu")
                echo "everything works fine."
                ;;
            "Cuda102")
                echo "everything works fine."
                ;;
            "Cuda112")
                echo "everything works fine."
                ;;
            "Cuda116")
                echo "everything works fine."
                ;;
            "Cuda117")
                echo "everything works fine."
                ;;
            "Cuda118")
                echo "everything works fine."
                ;;
            "Cuda120")
                echo "everything works fine."
                ;;
            *)
            echo "Cuda Version is incorrect!!!"
            ;;
        esac

        case ${python_version} in
            "Python37")
                echo "###############################"
                echo "###         WARNING         ###"
                echo "###############################"
                echo "# Python37 is no longer supported in the develop branch. Please choose Python 3.8 or higher versions. #"
                echo "###############################"
                ;;
            "Python38")
                echo "everything works fine."
                ;;
            "Python39")
                echo "everything works fine."
                ;;
            "Python310")
                echo "everything works fine."
                ;;
            *)
                echo "Python Version is incorrect!!!"
                ;;
        esac
    fi

}

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
                export WHELLINFO_EXITCODE=116
            fi
            ;;
        *)
            export WHELLINFO_EXITCODE=115
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
            export WHELLINFO_EXITCODE=115
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
            export WHELLINFO_EXITCODE=115
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
        "Inference")
            export WHELLINFO_EXITCODE=116
            ;;
        *)
            export WHELLINFO_EXITCODE=115
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
        "Inference")
            export WHELLINFO_EXITCODE=116
            ;;
        *)
            export WHELLINFO_EXITCODE=115
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
        "Inference")
            export WHELLINFO_EXITCODE=116
            ;;
        *)
            export WHELLINFO_EXITCODE=115
            ;;
    esac
}


# 安装包链接
function WheelUrlInfo(){
    cuda_version=$1
    python_version=$2
    # branch信息默认为Develop分支
    branch_info=${3:-"Develop"}
    infer_package=${4:-"OFF"}

    # 增加Manual模式，如果已经手动设置了安装包链接，不执行后续逻辑
    if [[ "${paddle_whl}" != "" ]];then
        cuda_version="Manual"
    fi
    WHELLINFO_EXITCODE=0

    if [[ "${branch_info}" == "Release" ]];then
        echo "The branch information you selected is [${branch_info}];"
    elif [[ "${branch_info}" == "Develop" ]];then
        echo "The branch information you selected is [${branch_info}];"
    else
        echo "The branch information input is wrong, you can choose to enter 'Develop' or 'Release';"
        WHELLINFO_EXITCODE=114
    fi

    case ${cuda_version} in
        "Manual")
            echo "Select Manual Mode!!!"
            ;;
        "Cpu")
            CpuPackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "OFF" ]];then
                CpuPackageUrlInfo ${branch_info} Inference
            fi
            ;;
        "Cuda102")
            Cu102PackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "OFF" ]];then
                Cu102PackageUrlInfo ${branch_info} Inference
            fi
            ;;
        "Cuda112")
            Cu112PackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "OFF" ]];then
                Cu112PackageUrlInfo ${branch_info} Inference
            fi
            ;;
        "Cuda116")
            Cu116PackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "OFF" ]];then
                Cu116PackageUrlInfo ${branch_info} Inference
            fi
            ;;
        "Cuda117")
            Cu117PackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "OFF" ]];then
                Cu117PackageUrlInfo ${branch_info} Inference
            fi
            ;;
        "Cuda118")
            Cu118PackageUrlInfo ${branch_info} ${python_version}
            if [[ "${infer_package}" != "OFF" ]];then
                Cu118PackageUrlInfo ${branch_info} Inference
            fi
            ;;
        *)
            WHELLINFO_EXITCODE=115
            ;;
        esac
    if [[ "$cuda_version" != "Manual" ]];then
        # 安装包选择的条件信息
        echo "==========================="
        echo "Install Packages Select Options as follows:"
        echo "- Branch Info: ${branch_info}"
        echo "- Cuda Version: ${cuda_version}"
        echo "- Python Version: ${package_version}"
        echo "- Inference Packages: ${infer_package}"

        if [[ "${WHELLINFO_EXITCODE}" == "114" ]];then
            echo "Branch Info is incorrect,please choose one from Develop or Release!!!"
            echo "EXITCODE:${WHELLINFO_EXITCODE}"
        elif [[ "${WHELLINFO_EXITCODE}" == "115" ]];then
            echo "Cuda Version is incorrect!!!"
        elif [[ "${WHELLINFO_EXITCODE}" == "115" ]];then
            echo "Python Version is incorrect,please choose from (Python37 Python38 Python39 Python310 Inference)"
        else
            echo "====InstallPackage Info is:===="
            echo "- paddle_whl:${paddle_whl}"
            echo 'Ps. You can get this wheel_url through "${paddle_whl}";'

            if [[ "${infer_package}" != "OFF" ]];then
                echo "- paddle_inference:${paddle_inference}"
                echo "- paddle_inference_c:${paddle_inference_c}"
                echo 'Ps. You can get this wheel_url through "${paddle_inference}" or "${paddle_inference_c}"'
            fi
        fi

        if [[ "${WHELLINFO_EXITCODE}" != "0" ]];then
            echo "EXITCODE:${WHELLINFO_EXITCODE}"
            exit ${WHELLINFO_EXITCODE}
        fi
    else
        echo "====InstallPackage Info is:===="
        echo "- paddle_whl:${paddle_whl}"
        echo 'Ps. You can get this wheel_url through "${paddle_whl}";'
    fi
}

function print_usage(){
    echo -e "\n${RED}Usage${NONE}:
    ${BOLD}${SCRIPT_NAME}${NONE} [OPTION]"

    echo -e "\n${RED}Options${NONE}:
    ${BLUE}get_docker_images${NONE}: Get Docker Images, You can use get_docker_images as follows command\n
    1. Get Centos Images: \n
        'source linux_env_info.sh' \n
        'linux_env_info_main get_docker_images Centos Cuda117'\n
    2. Get Ubuntu Images: \n
        'source linux_env_info.sh' \n
        'linux_env_info_main get_docker_images Ubuntu Cuda117' \n
    ${BLUE}get_wheel_url${NONE}: Get Packages Url,You can use get_wheel_url as follows command\n
    1. Only for Python Wheel Url: \n
        'source linux_env_info.sh' \n
        'linux_env_info_main get_wheel_url Cuda112 Python310 Develop' \n
    2. Get Python Wheel And Inference Packages: \n
        'source linux_env_info.sh' \n
        'linux_env_info_main get_wheel_url Cuda112 Python39 Develop ON' \n
    "
}


function linux_env_info_main() {
    local CMD=$1
    local args=("$@")
    case $CMD in
        get_docker_images)
            DockerImages "${args[@]:1}"
            ;;
        get_wheel_url)
            VersionExitJudgment "${args[@]:1}"
            WheelUrlInfo "${args[@]:1}"
            ;;
        *)
            print_usage
            ;;
        esac
}

linux_env_info_main
set +e
