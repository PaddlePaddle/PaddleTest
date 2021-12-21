#!/bin/bash

#git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
#export PYTHONPATH=$PWD/PaddleScience:$PYTHONPATH
#pip3.7 install -r PaddleScience/requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
#

cases=`find . -maxdepth 1 -name "test_*.py" | sort `
ignore=""

for file in ${cases}
do
echo ${file}
if [[ ${ignore} =~ ${file##*/} ]]; then

    echo "跳过"

else

    python3.7 -m pytest ${file}
fi
done
