#!/bin/bash

REPO=$1

docker pull registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-aarch64-gcc82
docker rm -f tipc-cann601-card23_zy
docker run -i --name tipc-cann601-card23_zy -v `pwd`:/workspace \
       -v /home/datasets:/datasets --workdir=/workspace \
       --pids-limit 409600 --network=host --shm-size=128G \
       --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
       --device=/dev/davinci2 \
       --device=/dev/davinci3 \
       --device=/dev/davinci_manager \
       --device=/dev/devmm_svm \
       --device=/dev/hisi_hdc \
       -e "repo=${REPO}" \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
       -v /usr/local/dcmi:/usr/local/dcmi \
       registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-aarch64-gcc82 /bin/bash -c -x "

#!/bin/bash

unset http_proxy
unset https_proxy

export repo=${REPO}

python -m pip install --retries 50 --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip config set global.index-url https://mirror.baidu.com/pypi/simple;
python -m pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set install.trusted-host mirror.baidu.com,pypi.tuna.tsinghua.edu.cn

# 2) 安装飞桨昇腾 nightly build 安装包 paddle-custom-npu
#wget https://paddle-device.bj.bcebos.com/develop/npu/paddle_custom_npu-0.0.1-cp37-cp37m-linux_aarch64.whl
#pip install -U https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp37-cp37m-linux_aarch64.whl
#pip install -U paddlepaddle-2.5.0-cp37-cp37m-linux_aarch64.whl
#pip install -U paddle_custom_npu-2.5.0-cp37-cp37m-linux_aarch64.whl

# pip install --force-reinstall --no-deps paddle_custom_npu-0.0.0-cp37-cp37m-linux_aarch64.whl
# pip install --force-reinstall --no-deps paddlepaddle-0.0.0-cp37-cp37m-linux_aarch64.whl

rm -rf paddlepaddle-*.whl
wget -q  https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp39-cp39-linux_aarch64.whl
pip install paddlepaddle-*.whl --force-reinstall --no-deps
rm -rf paddle_custom_npu*.whl
wget -q  https://paddle-device.bj.bcebos.com/develop/npu/paddle_custom_npu-0.0.0-cp39-cp39-linux_$(uname -m).whl
pip install paddle_custom_npu*.whl --force-reinstall 

python -c 'import paddle_custom_device; print(paddle_custom_device.npu.version())'
python -c 'import paddle; print(paddle.version.commit)'

#rm -rf AutoLog
#git clone --depth=100 https://github.com/LDOUBLEV/AutoLog
#cd ./AutoLog
#python -m pip install --retries 10 -r requirements.txt
#python setup.py bdist_wheel
#cd -
python -m pip install ./AutoLog/dist/*.whl

echo ${REPO}

bash test.sh ${REPO}
"
