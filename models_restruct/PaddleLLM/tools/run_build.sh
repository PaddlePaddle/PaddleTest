export https_proxy=${proxy}
export http_proxy=${proxy}
export root_path=$PWD

####    套件库下载    #####
wget -q --no-proxy  https://paddle-qa.bj.bcebos.com/CodeSync/develop/PaddleNLP.tar --no-check-certificate
rm -rf PaddleNLP && tar xf PaddleNLP.tar && rm -rf PaddleNLP.tar 
# cd PaddleNLP && git fetch origin pull/10596/head:PR_10596 && git checkout PR_10596 && cd -
# fix CUDA error 801 with PR 10570 in docker 
sed -i '/from transformer_engine import transformer_engine_paddle as tex/,/^    }/d' PaddleNLP/paddlenlp/quantization/qat_utils.py
unset http_proxy && unset https_proxy

# python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-TagBuild-Training-Linux-Gpu-Cuda12.8-Cudnn9.7-Trt10.5-Mkl-Avx-Gcc11-SelfBuiltPypiUse/3b5fe1f4e5b4bd71f1c0b8e33d459f2f4caff554/paddlepaddle_gpu-3.0.0.dev20250423-cp310-cp310-linux_x86_64.whl --force-reinstall --no-dependencies
####    for cuda12.8 pdc image    #####
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib/:${LD_LIBRARY_PATH}

####    执行    #####
today=$(date +%Y%m%d)
paddlenlp_ops_url="https://paddlenlp.bj.bcebos.com/wheels/paddlenlp_ops-3.0.0b4.post${today}+cuda12.8sm80paddle3b5fe1f-py3-none-any.whl"
if curl --output /dev/null --silent --head --fail "$paddlenlp_ops_url"; then
    echo "bos文件存在，可下载，退出当前编译任务"
    exit 0
else
    echo "bos文件不存在，前置执行构建命令"
    #python -m pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html --no-cache-dir
    cd PaddleNLP/csrc
    bash tools/build_wheel.sh > build.log 2>&1 
    echo "编译完成，当前环境已更新paddlenlp_ops，后续case执行无须安装"
    cp -r $root_path/PaddleNLP/csrc/gpu_dist/p****.whl /root/paddlejob/workspace/env_run/agent/workspace/
fi