#! /bin/bash
export http_proxy=http://172.19.57.45:3128
export https_proxy=http://172.19.57.45:3128
apt-get install python3-venv

cur_env=$VIRTUAL_ENV
new_env=`date  +%Y-%m-%d-%H-%M-%S`
echo $new_env

python -m venv $new_env
. ./${new_env}/bin/activate
pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

python tool.py --check_item $1

rm -rf $new_env
