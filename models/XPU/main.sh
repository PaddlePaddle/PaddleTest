#!/bin/bash

REPO=$1

docker pull registry.baidubce.com/device/paddle-xpu:ubuntu18-x86_64-gcc82
docker rm -f tipc-xpu-sjx
docker run -i --name tipc-xpu-sjx -v `pwd`:/workspace \
       -v /home/datasets:/datasets --workdir=/workspace \
       -w=/workspace --shm-size=128G --network=host --privileged  \
       --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
       -e "repo=${REPO}" \
       registry.baidubce.com/device/paddle-xpu:ubuntu18-x86_64-gcc82 /bin/bash -c -x "

#!/bin/bash

unset http_proxy
unset https_proxy

export repo=${REPO}

python -m pip install --retries 50 --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip config set global.index-url https://mirror.baidu.com/pypi/simple;
python -m pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set install.trusted-host mirror.baidu.com,pypi.tuna.tsinghua.edu.cn

rm -rf paddlepaddle_*.whl
wget -q  https://paddle-device.bj.bcebos.com/develop/xpu/paddlepaddle_xpu-0.0.0-cp39-cp39-linux_x86_64.whl
python -m pip install paddlepaddle_*.whl --force-reinstall 

python -c 'import paddle; print(paddle.version.commit)'
python -m pip install ./AutoLog/dist/*.whl
echo ${REPO}

bash run.sh ${REPO}
"

