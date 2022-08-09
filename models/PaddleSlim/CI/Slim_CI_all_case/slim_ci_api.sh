#!/usr/bin/env bash
#bash slim_ci_api.sh %python% %paddle% %http_proxy% ${cudaid1} ${cudaid2};"
echo "enter slim_ci_api.sh, params:" $1,$2,$3,$4,$5
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
echo -e "\033[35m ---- install_paddle: python=$1, paddle= $2   \033[0m"
python -m pip install --upgrade pip
python -m pip install $2 --no-cache-dir;
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
    python setup.py install
}
slim3_build_whl (){
    echo -e "\033[35m ---- build and install slim  \033[0m"
    python -m pip install matplotlib
    python -m pip install -r requirements.txt
    python setup.py bdist_wheel --universal
    python -m pip install dist/paddleslim-1.0.0-py2.py3-none-any.whl
}
yum -y install perl-Digest-MD5
python -m pip install matplotlib
python -m pip install -r requirements.txt
python -m pip install -r requirements_ci.txt
#$3   api coverage not install slim
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
git remote add upstream https://github.com/PaddlePaddle/PaddleSlim.git
git fetch upstream
##########################
IF_UT=false
for file_name in `git diff --numstat upstream/develop |awk '{print $NF}'`;do
    dir1=${file_name%%/*}
    echo ${file_name}
#    if [[ ${file_name##*.} =~ "md" ]] || [[ ${file_name##*.} =~ "rst" ]] || [[ ${dir1} =~ "demo" ]] || [[ ${dir1} =~ "docs" ]];then
    if [[ ${dir1} =~ "tests" ]] || [[ ${dir1} =~ "paddleslim" ]] ;then
        IF_UT=true
        break
    else
        continue
    fi
done
echo -e "\033[35m ---- IF_UT: $IF_UT  \033[0m"

#IF_UT=true
UT_EXCODE=0
if [ $IF_UT == 'true' ];then
    bash slim_ci_api_coverage.sh $4 $5;
    UT_EXCODE=$? || true
    echo -e "\033[35m ---- UT_EXCODE: $UT_EXCODE  \033[0m"
fi
##################
check_code_style(){
python -m pip install pip==20.2.4
pip install cpplint pylint pytest astroid isort
pip install pre-commit
pre-commit install
commit_files=on
check_sty_EXCODE=0

for file_name in `git diff --numstat upstream/develop |awk '{print $NF}'`;do
#for file_name in `git diff --numstat develop |awk '{print $NF}'`;do
    echo -e "\033[35m ---- checking for: $file_name \033[0m"
    if ! pre-commit run --files $file_name ; then
        echo -e "\033[31m ---- check fail file_name: $file_name \033[0m"
        git diff
        commit_files=off
    fi
done
if [ $commit_files == 'off' ];then
    echo -e "\033[31m ---- check code style fail  \033[0m"
    check_sty_EXCODE=2
fi
}
check_code_style || true
####################################
echo -e "\033[35m ---- result: \033[0m"
EXCODE=0
if [ $check_sty_EXCODE -eq 2 ];then
    echo -e "\033[31m ---- check code style Failed!  \033[0m"
    EXCODE=2
    exit $EXCODE
fi
if [ $UT_EXCODE -eq 0 ];then
    echo -e "\033[32m ---- unit test Success  \033[0m"
elif [ $UT_EXCODE -eq 1 ]; then
    echo -e "\033[31m ---- unit test Failed  \033[0m"
    cd ${slim_dir}/logs
    ls *_FAIL*
    exit $UT_EXCODE
elif [ $UT_EXCODE -eq 9 ]; then
    echo -e "\033[31m ---- Coverage Failed!  \033[0m"
    exit $UT_EXCODE
else
    echo -e "\033[31m ---- unit test Failed  \033[0m"
    exit $UT_EXCODE
fi
exit $EXCODE
