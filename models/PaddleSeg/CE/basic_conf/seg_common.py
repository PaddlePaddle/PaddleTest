# encoding: utf-8
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件提供了代码中用到的公共配置。
Creators: paddlepaddle-qa
Date:    2021/02/17 14:33:27
"""
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# 公共全局变量
PADDLE_ON_MODEL_CE = "1"
WITH_AVX = "ON"
# 资源配置
IS_SINGLE_CUDA = True
XPU = "gpu"  # 取值gpu或cpu
# 配置文件中SET_CUDA为None，框架不配置gpu cuda，使用cpu场景
SET_CUDA = "0"
SET_MULTI_CUDA = "0,1"
SET_MULTI_CUDA_CONVERGENCE = "0,1,2,3"
"""
if XPU == 'gpu':
    if IS_SINGLE_CUDA:
        SET_CUDA = '0'
    else:
        SET_MULTI_CUDA = '0,1'
"""
# PaddleSeg
REPO_PaddleSeg = "https://github.com/PaddlePaddle/PaddleSeg.git"
BASE_BRANCH = "develop"
# GPT2_BRANCH = BASE_BRANCH
# transformer_BRANCH = BASE_BRANCH
EXEC_TAG = [
    "linux_dy_gpu1",
    "linux_dy_gpu2",
    "linux_dy_gpu4",
    "linux_dy_gpu2_con",
    # 补充一些自定义标签
    "linux_dy_process_data",
]
