#!/usr/bin/env bash
# set python env
#nlp_ci.sh $1{python_version} $2{paddle_compile} $3{nlp_install_method} $4{http_proxy};
case $1 in
27)
  export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.15-ucs2/lib/:${LD_LIBRARY_PATH}
  export PATH=/opt/_internal/cpython-2.7.15-ucs2/bin/:${PATH}
  ;;
35)
  export LD_LIBRARY_PATH=/opt/_internal/cpython-3.5.1/lib/:${LD_LIBRARY_PATH}
  export PATH=/opt/_internal/cpython-3.5.1/bin/:${PATH}
  ;;
36)
  export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
  export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
  ;;
37)
  export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
  export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
  ;;
38)
  export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
  export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
  ;;
esac
python -c 'import sys; print(sys.version_info[:])'
echo "python="$1
####################################
set -x
python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
# set paddleslim env 
cd PaddleSlim 
python -m pip install -r requirements.txt
python setup.py install
cd -
####################################
# for paddlenlp env
nlp1_build (){
    echo -e "\033[35m ---- only install paddlenlp \033[0m"
    python -m pip install -U paddlenlp
}
nlp2_build (){
    echo -e "\033[35m ---- build and install paddlenlp  \033[0m"
    rm -rf build/
    rm -rf paddlenlp.egg-info/
    rm -rf dist/

    python -m pip install -r requirements.txt
    python setup.py bdist_wheel
    python -m pip install dist/paddlenlp****.whl
}
$3
python -m pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c 'from visualdl import LogWriter'
pip list
set +x
####################################
# for logs env
export nlp_dir=/workspace
if [ -d "/workspace/logs" ];then
    rm -rf /workspace/logs;
fi
mkdir /workspace/logs
export log_path=/workspace/logs
####################################
# run p0case
bash daily_case.sh ${cudaid1} ${cudaid2} ${Testcase}
P0case_EXCODE=$? || true
####################################
echo -e "\033[35m ---- result: \033[0m"
echo -e "\033[35m ---- P0case_EXCODE: $P0case_EXCODE \033[0m"
if [ $P0case_EXCODE -ne 0 ] ; then
    cd logs
    FF=`ls *_FAIL*|wc -l`
    echo -e "\033[31m ---- P0case failed number: ${FF} \033[0m"
    ls *_FAIL*
    exit $P0case_EXCODE
else
    echo -e "\033[32m ---- P0case Success \033[0m"
fi
