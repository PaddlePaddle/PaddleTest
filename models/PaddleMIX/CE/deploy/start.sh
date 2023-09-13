#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}

pip install fastdeploy-gpu-python==1.0.7 -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html  -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com

sh test_controlnet_infer_dygraph.sh 2>&1 | tee test_controlnet_infer_dygraph.log

sh test_infer_dygraph.sh 2>&1 | tee test_infer_dygraph.log

sh test_controlnet_infer_fd.sh 2>&1 | tee test_controlnet_infer_fd.log

sh test_infer_fd.sh 2>&1 | tee test_infer_fd.log