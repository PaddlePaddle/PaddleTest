#!/bin/bash

rm -rf data
wget https://paddlenlp.bj.bcebos.com/models/community/westfish/t2i-adapter/t2i-adapter-data-demo.zip
unzip -o t2i-adapter-data-demo.zip

export http_proxy=${proxy}
export https_proxy=${proxy}
pip install -r requirements.txt
unset http_proxy
unset https_proxy