#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}
pip install -U ppdiffusers visualdl
unset http_proxy
unset https_proxy