#!/usr/bin/env bash
export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
export python_install_path=/opt/_internal/cpython-3.7.0/lib/python3.7/site-packages
set +x
export no_proxy=bcebos.com
export http_proxy=${http_proxy}
export https_proxy=${http_proxy}
export CUDA_VISIBLE_DEVICES=${cudaid1}
set -x
python -m pip install --ignore-installed --upgrade pip
python -m pip install  ${paddle_compile}
python -m pip install  -r requirements.txt
unset http_proxy
unset https_proxy
ls
mkdir log
python -m pytest -sv test_paddlenlp_aistudio.py::test_aistudio_case --alluredir=./result
exit_code=$?
python gen_allure_report.py
exit exit_code