#!/bin/bash

work_path=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path}/

cd ${work_path}
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
