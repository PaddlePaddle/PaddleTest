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
Date:    2021/09/15 14:47:27
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
# if XPU == 'gpu':
#     if IS_SINGLE_CUDA:
#         SET_CUDA = '0'
#     else:
#         SET_CUDA = '0'
#         SET_MULTI_CUDA = '0,1'

# Parakeet
REPO_Parakeet = "https://github.com/PaddlePaddle/Parakeet.git"
BASE_BRANCH = "develop"
parakeet_BRANCH = "develop"
