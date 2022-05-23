#!/usr/bin/env bash
echo "enter run slim api, params:" $1,$2,$3,$4,$5

# set slim_dir and logs path
# workspace == PaddleSlim/
export slim_dir=/workspace
if [ -d "/workspace/logs" ];then
    rm -rf /workspace/logs;
fi
mkdir /workspace/logs
export log_path=/workspace/logs
  
#python version、paddle_compile_path、slim_install_method
bash slim_prepare_env.sh $1 $2 $3

# cudaid1、cudaid2
bash slim_ci_api_coverage.sh $4 $5;
UT_EXCODE=$? || true


check_code_style(){
python -m pip install pip==20.2.4
pip install cpplint pylint pytest astroid isort
pip install pre-commit
pre-commit install
commit_files=on
check_sty_EXCODE=0

for file_name in `git diff --numstat upstream/develop |awk '{print $NF}'`;do
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
echo --- run result---
EXCODE=0
if [ $check_sty_EXCODE -eq 2 ];then
    echo -e "\033[31m ---- check code style Failed!  \033[0m"
    EXCODE=2
fi
if [ $UT_EXCODE -eq 0 ];then
    echo -e "\033[32m ---- unit test Success  \033[0m"
elif [ $UT_EXCODE -eq 1 ]; then
    echo -e "\033[31m ---- unit test Failed  \033[0m"
    cd ${log_path}
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