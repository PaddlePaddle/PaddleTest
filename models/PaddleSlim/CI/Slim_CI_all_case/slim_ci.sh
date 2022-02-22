#!/usr/bin/env bash
#bash slim_ci.sh %python% %paddle% %paddleSlim% %http_proxy% ${cudaid1} ${cudaid2};
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
  export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
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
    python setup.py bdist_wheel
    python -m pip install dist/*.whl;
}
$3

yum -y install perl-Digest-MD5
python -m pip install -r requirements_ci.txt
pip list
####################################
# for logs env
export slim_dir=/workspace
export data_path=/paddle/all_data
if [ -d "/workspace/logs" ];then
    rm -rf /workspace/logs;
fi
mkdir /workspace/logs
export log_path=/workspace/logs
####################################
bash slim_ci_api_coverage.sh $5 $6;
UT_EXCODE=$? || true
echo -e "\033[35m ---- UT_EXCODE: $UT_EXCODE  \033[0m"
####################################
# run all case
bash slim_ci_demo_all_case.sh $5 $6;
P0case_EXCODE=$? || true
echo -e "\033[35m ---- P0case_EXCODE: $P0case_EXCODE  \033[0m"
####################################
# check RD who failed a unit test
cd /workspace
check_RD_paddleUT (){
    echo -e "\033[35m ---- check RD who failed a unit test,GIT_PR_ID: ${GIT_PR_ID} \033[0m"
    set -x
    wget -q https://paddle-ci.gz.bcebos.com/blk/block.txt
    curl https://paddle-ci.gz.bcebos.com/blk/check_ut.py | python - PaddleSlim
}
check_RD_paddleUT
check_RD_ut_EXCODE=$? || true
####################################
echo -e "\033[35m ---- result: \033[0m"
echo -e "\033[35m ---- P0case_EXCODE: $P0case_EXCODE \033[0m"
echo -e "\033[35m ---- check_RD_paddleUT: $check_RD_ut_EXCODE \033[0m"
if [ $P0case_EXCODE -ne 0 ] ; then
    cd logs
    FF=`ls *_FAIL*|wc -l`
    echo -e "\033[31m ---- P0case failed number: ${FF} \033[0m"
    ls *_FAIL*
    for i in `ls *_FAIL*` 
    do 
      echo -----fail log as follow:$i----------
      cat $i 
    done
    
    exit $P0case_EXCODE
else
    echo -e "\033[32m ---- P0case Success \033[0m"
fi

if [ $check_RD_ut_EXCODE -ne 0 ] ; then
    exit $check_RD_ut_EXCODE
else
    echo -e "\033[32m ---- check_RD Success \033[0m"
fi
