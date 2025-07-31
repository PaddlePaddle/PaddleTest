#!/usr/bin/env bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
ROOT_DIR=$PWD/
nlp_dir=${ROOT_DIR}/PaddleNLP
upload_path=${ROOT_DIR}/Bos/upload
mkdir -p ${upload_path}
paddle_whl=$1

########### 判断是否进行编译 ########### 
get_diff_case(){
    export FLAGS_build_enable=false
    cd ${nlp_dir}
    git diff --name-only HEAD~1 HEAD
    for file_name in `git diff --name-only HEAD~1 HEAD`;do
        arr_file_name=(${file_name//// })
        if [[ ${arr_file_name[0]} == "csrc" ]];then
            FLAGS_build_enable=true
        else
            continue
        fi
    done

    if [[ ${FLAGS_build_enable} == true ]];then
        continue
    else
        exit 0
    fi
}
# get_diff_case


# get python3.10
wget -q https://paddlenlp.bj.bcebos.com/PaddleNLP_CI/PaddleNLP-Build-paddlenlp_ops/set_env.tar.gz
mkdir -p ~/miniconda3/envs/set_env
tar -zxf set_env.tar.gz -C ~/miniconda3/envs/set_env
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -u -p /root/miniconda3
source /root/miniconda3/etc/profile.d/conda.sh
conda activate set_env
python -m pip config set global.index-url http://pip.baidu.com/root/baidu/+simple/
python -m pip config set install.trusted-host  pip.baidu.com

# set LD_LIBRARY_PATH for cuda12.8
# cuda_version=$(echo "$paddle_whl" | grep -oP 'Cuda\K[0-9.]+')
# if [[ "$cuda_version" == "12.8" ]]; then
#     export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib/:${LD_LIBRARY_PATH}
# fi

echo -e " ---- Install paddlepaddle-gpu  ----"
python -m pip install --user ${paddle_whl};
python -c "import paddle;print('paddle');print(paddle.__version__);print(paddle.version.show())"
echo " ---- Install paddlenlp_develop  ----"
python -m pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html --no-cache-dir
python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)"


cd ${nlp_dir}/csrc
bash tools/build_wheel.sh all
set +e
#python -c "import paddlenlp_ops"
#ll ${nlp_dir}/csrc/gpu_dist

# for https://www.paddlepaddle.org.cn/whl/paddlenlp.html
cp ${nlp_dir}/csrc/gpu_dist/p****.whl ${upload_path}/
python -m pip install bce-python-sdk==0.8.74 --trusted-host pip.baidu-int.com --force-reinstall
cd ${upload_path} && ls -A "${upload_path}"
cd ${ROOT_DIR}/Bos && python upload.py ${upload_path} 'paddlenlp/wheels'
rm -rf ${upload_path}
echo -e " ---- upload wheels SUCCESS  ----"
