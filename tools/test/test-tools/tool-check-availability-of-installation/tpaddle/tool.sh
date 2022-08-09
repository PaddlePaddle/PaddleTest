#! /bin/bash

apt-get install python3-venv
cur_env=$VIRTUAL_ENV
new_env=`date  +%Y-%m-%d-%H-%M-%S`
echo $new_env

python -m venv $new_env
. ./${new_env}/bin/activate
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

python tool.py --check_item $1


rm -rf $new_env
