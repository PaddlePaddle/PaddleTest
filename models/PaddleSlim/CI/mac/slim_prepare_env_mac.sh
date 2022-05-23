#!/usr/bin/env bash
#python version、paddle_compile_path、slim_install_method
echo ---slim prepare env -----
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
others)
  echo ---skip python version---
  ;;
esac
echo ---python version:$1---
python -c 'import sys; print(sys.version_info[:])'

####################################
# for paddle env
echo ---install paddle---
python -m pip uninstall paddlepaddle -y
python -m pip install --upgrade pip
python -m pip install $2 --no-cache-dir
echo ---paddle commit:---
python -c 'import paddle; print(paddle.version.commit)';

####################################
# # for logs env
# export slim_dir=$PWD/PaddleSlim;
# cd ${slim_dir}
# if [ -d "logs" ];then
#     rm -rf logs;
# fi
# mkdir logs
# export log_path=${slim_dir}/logs;
####################################
# for paddleslim env
echo ---start install paddleslim---
python -m pip uninstall paddleslim -y
cd ${slim_dir}
slim1_install (){
    python -m pip install -U paddleslim
}
slim2_build (){
    python -m pip install matplotlib
    python -m pip install -r requirements.txt
    python setup.py install
}
slim3_build_whl (){
    python -m pip install matplotlib
    python -m pip install -r requirements.txt
    python setup.py bdist_wheel
    python -m pip install dist/*.whl;
}
$3
echo ---installed paddleslim---
python -m pip list | grep paddleslim
