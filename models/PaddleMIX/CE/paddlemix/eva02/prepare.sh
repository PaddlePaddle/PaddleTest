#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}
cd paddlemix/external_ops/
python setup.py install
unset http_proxy
unset https_proxy