#指定docker镜像
if [[ ${AGILE_PIPELINE_NAME} =~ "Cuda102" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7.6-trt7.0-gcc8.2"
    else
        Image_version="registry.baidubce.com/paddlepaddle/paddleqa:latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2"
        #230320 change registry.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7-dev for add trt
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda112" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.2-cudnn8.1-trt8.0-gcc8.2"
    else
        Image_version="registry.baidubce.com/paddlepaddle/paddleqa:latest-dev-cuda11.2-cudnn8.2-trt8.0-gcc82"
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda116" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.6-cudnn8.4.0-trt8.4.0.6-gcc82"
    else

        Image_version="registry.baidubce.com/paddlepaddle/paddleqa:latest-dev-cuda11.6.2-cudnn8.4.0-trt8.4.0.6-gcc82"
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda117" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        Image_version="registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.7-cudnn8.4-trt8.4-gcc8.2"
    else
        Image_version="registry.baidubce.com/paddlepaddle/paddleqa:latest-dev-cuda11.7-cudnn8.4-trt8.4-gcc8.2-v1"
    fi
else
    Image_version="registry.baidubce.com/paddlepaddle/paddleqa:latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2"
fi

echo "Image_version: ${Image_version}"
touch models_docker
echo "FROM ${Image_version}"> ./models_docker
md5sum models_docker |cut -d' ' -f1|xargs echo md5=
