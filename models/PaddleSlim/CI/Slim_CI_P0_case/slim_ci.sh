#!/usr/bin/env bash
echo "P0case_list:" ${P0case_list[*]}
echo "enter slim_ci.sh, params:" $1,$2,$3,$4,$5,$6
# set python env
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
  export PATH=/opt/_internal/cpython-3.8.0/bin/::${PATH}
  ;;
esac
python -c 'import sys; print(sys.version_info[:])'
echo "python="$1
####################################
# for paddle env
set -x
echo -e "\033[35m ---- install paddle: python=$1, paddle= $2   \033[0m"
python -m pip install --upgrade pip
python -m pip install $2;
python -c 'import paddle; print(paddle.version.commit)';
set +x;
####################################
# for paddleslim env
slim1_install (){
    echo -e "\033[35m ---- only install slim \033[0m"
    python -m pip install -U paddleslim
}
slim2_build (){
    echo -e "\033[35m ---- build and install slim  \033[0m"
    python -m pip install matplotlib
    python -m pip install -r requirements.txt
    python setup.py install
}
slim3_build_whl (){
    echo -e "\033[35m ---- build and install slim  \033[0m"
    python -m pip install matplotlib
    python -m pip install -r requirements.txt
    python setup.py bdist_wheel --universal
    python -m pip install dist/*.whl
}
$3
python -m pip install -r requirements_ci.txt
pip list
set +x
####################################
# for logs env
export slim_dir=/workspace
if [ -d "/workspace/logs" ];then
    rm -rf /workspace/logs;
fi
mkdir /workspace/logs
export log_path=/workspace/logs
####################################

# run p0case
bash slim_ci_p0case.sh $5 $6;
P0case_EXCODE=$? || true
####################################
echo -e "\033[35m ---- result: \033[0m"
echo -e "\033[35m ---- P0case_EXCODE: $P0case_EXCODE \033[0m"
if [ $P0case_EXCODE -ne 0 ] ; then
    cd logs
    FF=`ls *FAIL*|wc -l`
    echo -e "\033[31m ---- P0case failed number: ${FF} \033[0m"
    ls *FAIL*
    exit $P0case_EXCODE
else
    echo -e "\033[32m ---- P0case Success \033[0m"
fi
