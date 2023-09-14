#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}
pip install -U ppdiffusers visualdl -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com
unset http_proxy
unset https_proxy