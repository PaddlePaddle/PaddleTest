#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}
pip install -r requirements.txt
unset http_proxy
unset https_proxy