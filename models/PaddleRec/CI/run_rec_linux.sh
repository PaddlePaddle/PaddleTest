#! /bin/bash
# $1:python_version $2:paddle_compile_path $3:run_CI/run_CE/run_ALL/run_CPU  $4:cudaid1 $5:cudaid2

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
esac
python -c 'import sys; print(sys.version_info[:])';

echo ---install paddle---
python -m pip install $2 --no-cache-dir
echo ---paddle commit---
python -c 'import paddle; print(paddle.version.commit)';
python -c 'import paddle; paddle.utils.run_check()';

python -m pip install opencv-python
python -m pip install pandas
python -m pip install sklearn
python -m pip install scipy
python -m pip install tools
python -m pip install numba
python -m pip install pgl
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

# run_CI/run_CE/run_ALL/run_CPU 、cudaid1、cudaid2
bash rec_run_case_linux.sh $3 $4 $5


cd ${log_path}
FF=`ls *FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    echo ---fail case: ${FF}
    ls *FAIL*
    exit 1
else
    echo ---all case pass---
    exit 0
fi
