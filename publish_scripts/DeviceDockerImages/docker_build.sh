#!/bin/bash
PaddleVersion=$2

case "$1" in
  "xpu")
    PRODUCT_NAME="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:${PaddleVersion}-xpu-ubuntu20-x86_64-gcc84-py310"
    dockerfile=Dockerfile.xpu-p800
    ;;
  "dcu")
    PRODUCT_NAME="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-dcu:${PaddleVersion}-dtk24.04.1-kylinv10-gcc82-py310"
    dockerfile=Dockerfile.dcu
    ;;
  "npu-x86")
    PRODUCT_NAME="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:${PaddleVersion}-cann800-ubuntu20-npu-910b-x86_64-gcc84-py310"
    dockerfile=Dockerfile.npu-x86
    ;;
  "npu-aarch")
    PRODUCT_NAME="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:${PaddleVersion}-cann800-ubuntu20-npu-910b-aarch64-gcc84-py310"
    dockerfile=Dockerfile.npu-aarch
    ;;
  "mlu")
    PRODUCT_NAME="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-mlu:${PaddleVersion}-ctr2.15.0-ubuntu20-gcc84-py310"
    dockerfile=Dockerfile.mlu
    ;;
  "gcu")
    PRODUCT_NAME="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:${PaddleVersion}-topsrider3.2.109-ubuntu20-x86_64-gcc84-py310"
    dockerfile=Dockerfile.gcu 
    ;;
  *)
    echo "Usage: $0 {xpu|dcu|npu-x86|npu-aarch|mlu|gcu}"
    exit 1
    ;;
esac

# 代理地址
docker build -t ${PRODUCT_NAME} -f ${dockerfile} . \
    --network host \
    --build-arg PADDLE_VERSION=${PaddleVersion}
    # --build-arg HTTP_PROXY=${proxy} \
    # --build-arg HTTPS_PROXY=${proxy} \
    # --build-arg ftp_proxy=${proxy}

docker push ${PRODUCT_NAME}
