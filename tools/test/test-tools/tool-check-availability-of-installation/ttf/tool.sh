#! /bin/bash

apt-get install python3-venv
apt-get install python3.8-venv

cur_env=$VIRTUAL_ENV
new_env=`date  +%Y-%m-%d-%H-%M-%S`
echo $new_env

python3 -m venv --system-site-packages $new_env
. ./${new_env}/bin/activate
pip install --upgrade tensorflow

python tool.py --check_item $1

rm -rf $new_env
