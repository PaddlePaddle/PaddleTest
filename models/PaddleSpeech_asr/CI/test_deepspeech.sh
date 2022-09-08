#!/bin/bash
set -e
set -x
mkdir run_env_py37;
ln -s $(which python3.7) run_env_py37/python;
ln -s $(which pip3.7) run_env_py37/pip;
export PATH=$(pwd)/run_env_py37:${PATH};
export http_proxy=${proxy}
export https_proxy=${proxy}
export no_proxy=bcebos.com;
apt-get update
apt-get install -y sox pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig python3-dev
pushd tools; make virtualenv.done; popd
if [ $? -ne 0 ];then
    exit 1
fi
source tools/venv/bin/activate
python -m pip install pip --ignore-installed;
# python -m pip install ${paddle_whl} --no-cache-dir
 python -m pip install paddlepaddle-gpu==2.3.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -e .
cd dataset
rm -rf librispeech
ln -s /ssd2/ce_data/PaddleSpeech_t2s/preprocess_data/deepspeech/librispeech librispeech
cd ..
cd examples/tiny/asr0
source path.sh
export CUDA_VISIBLE_DEVICES=$cudaid2
#create log dir
if [ -d "log" ];then rm -rf log
fi
mkdir log
if [ -d "log_err" ];then rm -rf log_err
fi
mkdir log_err
sed -i 's/CUDA_VISIBLE_DEVICES=${gpus}//g' run.sh
bash run.sh >log/run.log 2>&1
if [ $? -ne 0 ];then
    echo -e "tiny/s0, FAIL"
    mv log/run.log log_err/
    err_sign=true
else
    echo -e "tiny/s0, SUCCESS"
fi
if [ "${err_sign}" = true ];then
    exit 1
fi
set +x
