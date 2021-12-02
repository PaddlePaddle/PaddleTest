#!/bin/bash
set -e
set -x
mkdir run_env_py37;
ln -s $(which python3.7) run_env_py37/python;
ln -s $(which pip3.7) run_env_py37/pip;
export PATH=$(pwd)/run_env_py37:${PATH};
python -m pip install pip==20.2.4 --ignore-installed;
export no_proxy=bcebos.com;
python -m pip install ${paddle_whl} --no-cache-dir
export http_proxy=${proxy}
export https_proxy=${proxy}
apt-get update
apt-get install -y sox pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig python3-dev
pushd tools; make virtualenv.done; popd
if [ $? -ne 0 ];then
    exit 1
fi
source tools/venv/bin/activate
python -m pip install pip==20.2.4 --ignore-installed;
python -m pip install ${paddle_whl} --no-cache-dir
python -m pip install numpy==1.20.1 --ignore-installed
python -m pip install pyparsing==2.4.7 --ignore-installed
pip install -e .
cd dataset
rm -rf librispeech
ln -s /ssd1/xiege/data/librispeech librispeech
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
