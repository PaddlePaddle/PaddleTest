#! /bin/bash
# $1:python_version $2:paddle_compile_path $3:run_CI/run_CE/run_ALL/run_CPU $4:cudaid1 $5:cudaid2
echo "---$1、$2、$3、$4--"

case $1 in
27)
  export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs2/lib/:${LD_LIBRARY_PATH}
  export PATH=/opt/_internal/cpython-2.7.11-ucs2/bin/:${PATH}
  ;;
36)
  export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
  export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
  ;;
37)
  export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
  export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
  ;;
'others')
  export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
  export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
  ;;
esac
python -c 'import sys; print(sys.version_info[:])';

echo ---install paddle---
python -m pip uninstall paddlepaddle-gpu -y
python -m pip uninstall paddlepaddle -y
python -m pip install $2 --no-cache-dir
echo ---paddle commit---
python -c 'import paddle; print(paddle.version.commit)';
python -c 'import paddle; paddle.utils.run_check()';

python -m pip install opencv-python
python -m pip install pandas
python -m pip install sklearn
python -m pip install scipy
python -m pip install numba
python -m pip install pgl
python -m pip install tqdm
python -m pip install pyyaml
python -m pip install requests
echo ---pip list---
python -m pip list

# set rec workdir
# workspace == PaddleRec ?
export rec_dir=/workspace/PaddleRec

# set logs path
if [ -d "/workspace/logs" ];then
    rm -rf /workspace/logs;
fi
mkdir /workspace/logs
export log_path=/workspace/logs

cd ${rec_dir}
git log --pretty=oneline -10
python -m pip install pip==20.2.4
python -m pip install cpplint pylint pytest astroid isort
python -m pip install pre-commit==2.9.3
pre-commit install

echo "---start code-style check---"
commit_files=on
git remote add upstream https://github.com/PaddlePaddle/PaddleRec.git
git fetch upstream
git diff --numstat upstream/master
git diff --numstat upstream/master |awk '{print $NF}'
for file_name in `git diff --numstat upstream/master |awk '{print $NF}'`;do
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
    exit 2;
fi

# run_CI/run_CE/run_ALL/run_CPU 、cudaid1、cudaid2
bash rec_run_case_linux.sh $3 $4 $5
exit $?

cd ${log_path}
FF=`ls *FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    echo ---fail case: ${FF}
else
    echo ---all case pass---
fi
