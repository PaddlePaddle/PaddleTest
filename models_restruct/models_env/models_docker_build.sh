if [ -e linux_env_info.sh ];then
    rm -rf linux_env_info.sh
fi
wget -q https://raw.githubusercontent.com/PaddlePaddle/PaddleTest/develop/tools/linux_env_info.sh
source ./linux_env_info.sh
set +e
#指定docker镜像
if [[ ${AGILE_PIPELINE_NAME} =~ "Cuda102" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        linux_env_info_main get_docker_images Centos Cuda102
    else
        linux_env_info_main get_docker_images Ubuntu Cuda102
        #230320 change registry.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7-dev for add trt
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda112" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        linux_env_info_main get_docker_images Centos Cuda112
    else
        linux_env_info_main get_docker_images Ubuntu Cuda112
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda116" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        linux_env_info_main get_docker_images Centos Cuda116
    else
        linux_env_info_main get_docker_images Ubuntu Cuda116
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda117" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        linux_env_info_main get_docker_images Centos Cuda117
    else
        linux_env_info_main get_docker_images Ubuntu Cuda117
    fi
else
    Image_version="registry.baidubce.com/paddlepaddle/paddleqa:latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2"
fi

echo "Image_version: ${Image_version}"
touch models_docker
echo "FROM ${Image_version}"> ./models_docker
md5sum models_docker |cut -d' ' -f1|xargs echo md5=
