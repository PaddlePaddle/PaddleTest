#!/bin/bash

cd ${root_path}/PaddleMIX
pip install -r requirements.txt
python -m pip install --upgrade pip
pip install -e .
pip install -r paddlemix/appflow/requirements.txt

cd ppdiffusers
pip install -e .